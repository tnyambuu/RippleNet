import tensorflow as tf
import numpy as np
from model import RippleNet
import os
import math
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from neo4j_connection import db_connection, verify_database, close_connection


def precision_at_k(true_positives, top_k_items, k):
    """Calculates Precision@K."""
    if not top_k_items:
        return 0.0
    hits = len(true_positives.intersection(set(top_k_items)))

    # print("true_positives: ", true_positives.intersection(set(top_k_items)))
    # print("top_k_items: ", top_k_items)
    # print("k: ", k)

    # print("hit: ", hits)
    return hits / k

def recall_at_k(true_positives, top_k_items, k):
    """Calculates Recall@K."""
    if not true_positives: # Handle case where user has no positive items in test set
        return 0.0 # Or 1.0 if you consider it trivially fulfilled, but 0.0 is safer
    if not top_k_items:
        return 0.0
    hits = len(true_positives.intersection(set(top_k_items)))
    return hits / len(true_positives)


def f1_at_k(precision_k, recall_k):
    """Calculates F1@K from Precision@K and Recall@K."""
    # Harmonic mean: Returns 0 if either P or R is 0
    if precision_k + recall_k == 0:
        return 0.0
    return 2 * (precision_k * recall_k) / (precision_k + recall_k)

def ndcg_at_k(true_positives, top_k_items_with_scores, k):
    """Calculates NDCG@K."""
    dcg = 0.0
    idcg = 0.0

    # Calculate DCG
    for i, (item_id, score) in enumerate(top_k_items_with_scores):
        if i >= k: # Only consider top K
            break
        if item_id in true_positives:
            # Relevance is 1 if it's a true positive, 0 otherwise
            relevance = 1
            # Discount factor: 1 / log2(rank + 1), rank starts at 1
            dcg += relevance / math.log2(i + 1 + 1) # Use i+1 for rank, add 1 for log base

    # Calculate IDCG (Ideal DCG)
    num_true_positives = len(true_positives)
    ideal_top_k = min(k, num_true_positives) # Can't have more relevant items than exist
    for i in range(ideal_top_k):
        idcg += 1 / math.log2(i + 1 + 1) # Ideal ranking has relevance 1 at top ranks

    return dcg / idcg if idcg > 0 else 0.0


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    print("n_entity: ", n_entity)
    print("n_relation: ", n_relation)
    print("rippleSet: ", len(ripple_set))

    model = RippleNet(args, n_entity, n_relation)

    # --- Add this section ---
    saver = tf.train.Saver(max_to_keep=5) # Create a Saver object

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path_prefix = os.path.join(args.save_dir, 'model') # Base path for checkpoints

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                # print("train------------->", ripple_set)
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            print("\n--- Evaluating Training Set ---")
            train_auc, train_acc, train_p, train_r, train_f1, train_ndcg = evaluation(sess, args, model, train_data, ripple_set, args.batch_size, k_value=100)
            print("\n--- Evaluating Evaluation Set ---")
            eval_auc, eval_acc, eval_p, eval_r, eval_f1, eval_ndcg = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size, k_value=100)
            print("\n--- Evaluating Test Set ---")
            test_auc, test_acc, test_p, test_r, test_f1, test_ndcg = evaluation(sess, args, model, test_data, ripple_set, args.batch_size, k_value=100)

            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            print(f'\n--- Epoch {step:d} Results ---')
            print(f'TRAIN\tAUC: {train_auc:.4f} ACC: {train_acc:.4f} P@10: {train_p:.4f} R@10: {train_r:.4f} F1@10: {train_f1:.4f} NDCG@10: {train_ndcg:.4f}')
            print(f'EVAL\tAUC: {eval_auc:.4f} ACC: {eval_acc:.4f} P@10: {eval_p:.4f} R@10: {eval_r:.4f} F1@10: {eval_f1:.4f} NDCG@10: {eval_ndcg:.4f}')
            print(f'TEST\tAUC: {test_auc:.4f} ACC: {test_acc:.4f} P@10: {test_p:.4f} R@10: {test_r:.4f} F1@10: {test_f1:.4f} NDCG@10: {test_ndcg:.4f}\n')

            if (step + 1) % args.save_period == 0 or step == args.n_epoch - 1: # Save periodically or on the last epoch
                print(f"Epoch {(step + 1)}: Saving model checkpoint...")
                checkpoint_path = saver.save(sess, save_path_prefix, global_step=step + 1)
                print(f"Model checkpoint saved to {checkpoint_path}")


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]

    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size, k_value):
    """
    Evaluates the model on the given dataset using AUC, Accuracy,
    and ranking metrics Precision@K, Recall@K, NDCG@K.
    """
    print(f"\nStarting evaluation for K={k_value}...")

    # --- 1. Group data by user ---
    user_item_ratings = defaultdict(list)
    unique_users = sorted(list(set(data[:, 0]))) # Get unique user IDs
    for i in range(data.shape[0]):
        user, item, label = data[i]
        # if (user == 20):
            # print("user: ", item)
        user_item_ratings[user].append((item, label))

    # print(unique_users)

    if not unique_users:
        print("Warning: No users found in the evaluation data.")
        return 0.0, 0.0, 0.0, 0.0, 0.0 # Return default values

    # --- 2. Initialize lists to store results ---
    all_true_labels = []
    all_pred_scores = []
    precision_list = []
    recall_list = []
    f1_list = []
    ndcg_list = []

    processed_users = 0

    # --- 3. Iterate through each user ---
    for user_id in unique_users:
        # Get items and labels for this user from the evaluation data
        user_data = user_item_ratings[user_id]
        if not user_data: continue # Skip if user somehow has no items in this set

        items_for_user = np.array([d[0] for d in user_data])
        # print(user_data)
        labels_for_user = np.array([d[1] for d in user_data])

        # Identify true positive items for this user IN THIS DATASET SPLIT
        true_positive_items = set(items_for_user[labels_for_user == 1])

        # Skip user if they have no positive items in this set for Recall/NDCG calculation?
        # Or allow calculation (Recall will be 0, NDCG depends on hits in top K)?
        # Let's allow calculation for now.

        # --- 4. Prepare feed_dict for all items of this user ---
        # We need the user's ripple set replicated for each item
        num_items_for_user = len(items_for_user)
        user_feed_data = np.zeros((num_items_for_user, 3), dtype=np.int32)
        user_feed_data[:, 0] = user_id         # Repeat user_id
        user_feed_data[:, 1] = items_for_user  # Use items for this user
        user_feed_data[:, 2] = labels_for_user # Use labels for this user

        # Use get_feed_dict, assuming it handles replicating ripple set based on user ID
        # Note: get_feed_dict might be inefficient if called repeatedly inside loop.
        # Let's create the feed dict directly here for clarity for this user.
        feed_dict = {
            model.items: items_for_user,
            model.labels: labels_for_user,
        }
        # Get user's ripple set data (assuming user_id is valid index/key)
        user_ripple_data = None
        if isinstance(ripple_set, dict):
            user_ripple_data = ripple_set.get(user_id)
        elif isinstance(ripple_set, list) and user_id < len(ripple_set):
            user_ripple_data = ripple_set[user_id]

        if not user_ripple_data or len(user_ripple_data) < args.n_hop:
            print(f"Warning: Insufficient ripple data for user {user_id}. Skipping user.")
            continue # Skip this user if ripple set isn't complete

        for i in range(args.n_hop):
            feed_dict[model.memories_h[i]] = [user_ripple_data[i][0]] * num_items_for_user
            feed_dict[model.memories_r[i]] = [user_ripple_data[i][1]] * num_items_for_user
            feed_dict[model.memories_t[i]] = [user_ripple_data[i][2]] * num_items_for_user


        # --- 5. Get prediction scores for this user's items ---
        try:
            pred_scores = sess.run(model.scores_normalized, feed_dict=feed_dict)
        except Exception as e:
            print(f"Error running session for user {user_id}: {e}. Skipping user.")
            continue

        # Handle scalar output just in case batch size was 1 somehow
        if not isinstance(pred_scores, np.ndarray):
            pred_scores = np.array([pred_scores])
        elif pred_scores.ndim == 0:
            pred_scores = np.array([pred_scores.item()])

        if len(pred_scores) != num_items_for_user:
            print(f"Warning: Score length mismatch for user {user_id} ({len(pred_scores)} vs {num_items_for_user}). Skipping user.")
            continue

        # Store for global AUC/Acc calculation
        all_true_labels.extend(labels_for_user.tolist())
        all_pred_scores.extend(pred_scores.tolist())

        # --- 6. Calculate Rank-based Metrics for this user ---
        # Combine items, scores, and labels
        item_score_label_list = list(zip(items_for_user, pred_scores, labels_for_user))

        # Sort by score descending
        item_score_label_list.sort(key=lambda x: x[1], reverse=True)

        # Get top K items (IDs only)
        top_k_items = [item_id for item_id, score, label in item_score_label_list[:k_value]]
        # Get top K items with scores (for NDCG)
        top_k_items_with_scores = [(item_id, score) for item_id, score, label in item_score_label_list[:k_value]]


        # Calculate metrics

        # print(f"User {user_id}")
        precision_k = precision_at_k(true_positive_items, top_k_items, k_value)
        recall_k = recall_at_k(true_positive_items, top_k_items, k_value)
        f1_k = f1_at_k(precision_k, recall_k)


        precision_list.append(precision_k)
        recall_list.append(recall_k)
        f1_list.append(f1_k)
        ndcg_list.append(ndcg_at_k(true_positive_items, top_k_items_with_scores, k_value))

        processed_users += 1
        if processed_users % 100 == 0:
            print(f"Evaluated {processed_users}/{len(unique_users)} users...", end='\r')


    print(f"\nFinished evaluation for {processed_users} users.")

    # --- 7. Calculate Final Average Metrics ---
    # Global AUC/Acc (more accurate than averaging batches)
    final_auc = 0.0
    final_acc = 0.0
    if all_true_labels and all_pred_scores:
        all_true_labels = np.array(all_true_labels)
        all_pred_scores = np.array(all_pred_scores)
        try:
            final_auc = roc_auc_score(all_true_labels, all_pred_scores)
        except ValueError:
            print("Warning: AUC calculation failed (likely only one class present).")
            final_auc = 0.0 # Or handle as needed
        final_predictions = (all_pred_scores >= 0.5).astype(int)
        final_acc = accuracy_score(all_true_labels, final_predictions)

    # Average Rank Metrics
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0


    # driver, session = db_connection("neo4j")
    # verify_database(session)

    # item_to_old = dict()
    # user_to_old = dict()

    # with open("../data/tender_v1/KG_item_to_id.txt", "r") as f:
    #     for line in f:
    #         line = line.split("\t")

    #         old_id = int(line[0])
    #         new_id = int(line[1])

    #         item_to_old[new_id] = old_id


    # with open("../data/tender_v1/KG_user_to_id.txt", "r") as f:
    #     for line in f:
    #         line = line.split("\t")

    #         old_id = int(line[0])
    #         new_id = int(line[1])

    #         user_to_old[new_id] = old_id

    # # top_n_items = heapq.nlargest(n_recommendations, item_scores.items(), key=lambda item: item[1])

    # print(f"\nTop {k_value} recommended items for user_id {user_id} user_old_id {user_to_old[user_id]}:")

    # old_user_id = user_to_old[user_id]
    # result_user = session.run(f"MATCH (n:Оролцогч)<-[r:АВАХ]-(y:ҮйлАжиллагааныЧиглэл) WHERE id(n) = {old_user_id} RETURN n.нэр as name, y.нэр as type")

    # user_name = None
    # type_name = None

    # for line in result_user:
    #     user_name = line["name"]
    #     type_name = line["type"]

    # print(f"\n  User ID: {user_id}, User name: {user_name} Type name: {type_name} \n")

    # for item_id, score in top_k_items_with_scores:
    #     old_item_id = item_to_old[item_id]
    #     result = session.run(f"MATCH (n:Урилга) WHERE id(n) = {old_item_id} RETURN n.нэр as name")

    #     tender_name = None
    #     for line in result:
    #         tender_name = line["name"]


    #     print(f"\n  Item ID: {item_id}, Old ID: {old_item_id}, Tender name: {tender_name} \n")

    # close_connection(driver, session)

    # Return all metrics
    return final_auc, final_acc, avg_precision, avg_recall, avg_f1, avg_ndcg
