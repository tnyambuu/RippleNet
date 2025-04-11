import tensorflow as tf
import numpy as np
import argparse
import os
from model import RippleNet
from data_loader import load_data # Or just load necessary parts
import heapq # For efficiently getting top N items

def predict_top_n_for_user(args, data_info, checkpoint_dir, user_id, n_recommendations=10):
    """Loads a saved model and predicts top N items for a given user."""

    # 1. Load necessary info
    n_item = data_info[6] # Get the total number of items from data_info
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    print(f"\nPredicting top {n_recommendations} items for user_id: {user_id}")

    # --- Get User's Ripple Set ---
    user_memories_h = []
    user_memories_r = []
    user_memories_t = []
    if user_id < len(ripple_set):
        for i in range(args.n_hop):
            # Ensure correct padding/structure if needed based on how ripple_set is built
            user_memories_h.append(ripple_set[user_id][i][0])
            user_memories_r.append(ripple_set[user_id][i][1])
            user_memories_t.append(ripple_set[user_id][i][2])
    else:
        print(f"Error: User ID {user_id} not found in ripple_set.")
        return None # Indicate failure

    # --- Define Candidate Items ---
    # Option 1: Predict for ALL items (can be slow for large N_item)
    candidate_items = list(range(n_item))

    # Option 2: Predict for items the user hasn't interacted with positively
    # (Requires loading user history)
    # user_history_dict = load_user_history(args) # Need a function for this
    # positive_items = set(user_history_dict.get(user_id, []))
    # candidate_items = [item for item in range(n_item) if item not in positive_items]

    # Option 3: Predict for a specific subset if needed

    if not candidate_items:
        print("No candidate items to predict for.")
        return []

    # --- Build Model and Restore ---
    tf.reset_default_graph()
    print("Building model graph...")
    model = RippleNet(args, n_entity, n_relation)
    print("Model graph built.")
    saver = tf.train.Saver()

    item_scores = {} # Dictionary to store {item_id: score}

    with tf.Session() as sess:
        # Restore model (same logic as before)
        print(f"Attempting to restore model from: {checkpoint_dir}")
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            checkpoint_path = ckpt_state.model_checkpoint_path
            try:
                saver.restore(sess, checkpoint_path)
                print(f"Model restored successfully from {checkpoint_path}")
            except Exception as e:
                print(f"Error restoring model: {e}")
                return None
        else:
            print(f"No checkpoint found in directory: {checkpoint_dir}")
            return None

        # --- Predict for items in batches ---
        batch_size = args.batch_size # Use the batch size argument
        start_index = 0
        while start_index < len(candidate_items):
            end_index = min(start_index + batch_size, len(candidate_items))
            batch_item_ids = candidate_items[start_index:end_index]

            # Prepare feed dict for the batch
            feed_dict = {
                model.items: batch_item_ids,
                model.labels: [0] * len(batch_item_ids) # Dummy labels
            }
            # Add user's ripple set data - REPLICATE for each item in batch
            for i in range(args.n_hop):
                feed_dict[model.memories_h[i]] = [user_memories_h[i]] * len(batch_item_ids)
                feed_dict[model.memories_r[i]] = [user_memories_r[i]] * len(batch_item_ids)
                feed_dict[model.memories_t[i]] = [user_memories_t[i]] * len(batch_item_ids)

            # Run inference for the batch
            # Use model.scores_normalized or model.scores based on previous debugging
            batch_scores = sess.run(model.scores_normalized, feed_dict=feed_dict) # Assuming this gives probabilities

            # Handle potential scalar output if batch size is 1
            if not isinstance(batch_scores, np.ndarray):
                batch_scores = np.array([batch_scores]) # Convert scalar to array

            # Store scores for items in this batch
            for item_id, score in zip(batch_item_ids, batch_scores):
                item_scores[item_id] = score

            start_index += batch_size
            print(f"Processed {start_index}/{len(candidate_items)} items...", end='\r')
        print("\nPrediction finished.")


    # --- Get Top N Recommendations ---
    if not item_scores:
        print("No scores were calculated.")
        return []

    # Use heapq for efficiency with large number of items
    # Finds the N items with the largest scores
    top_n_items = heapq.nlargest(n_recommendations, item_scores.items(), key=lambda item: item[1])
    # top_n_items will be a list of (item_id, score) tuples

    print(f"\nTop {n_recommendations} recommended items for user {user_id}:")
    for item_id, score in top_n_items:
        print(f"  Item ID: {item_id}, Score: {score:.4f}")

    return top_n_items


def predict(args, data_info, checkpoint_dir):
    """Loads a saved model and performs prediction."""

    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    # --- Prepare Input Data for Prediction ---

    user_id_to_predict = 0
    item_id_to_predict = 10

    # You need the ripple set for this user
    user_memories_h = []
    user_memories_r = []
    user_memories_t = []

    if user_id_to_predict < len(ripple_set): # Check if user exists
        for i in range(args.n_hop):
            user_memories_h.append(ripple_set[user_id_to_predict][i][0])
            user_memories_r.append(ripple_set[user_id_to_predict][i][1])
            user_memories_t.append(ripple_set[user_id_to_predict][i][2])
    else:
        print(f"Error: User ID {user_id_to_predict} not found in ripple_set.")
        return

    tf.reset_default_graph()

    print("Building model graph...")
    model = RippleNet(args, n_entity, n_relation)
    print("Model graph built.")

    # 3. Create Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 4. Restore Variables from Checkpoint
        print(f"Attempting to restore model from: {checkpoint_dir}")
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt_state and ckpt_state.model_checkpoint_path:
            # Restore from the latest checkpoint
            checkpoint_path = ckpt_state.model_checkpoint_path
            # OR specify a specific checkpoint:
            # checkpoint_path = os.path.join(checkpoint_dir, 'model-XX') # Replace XX with step number
            try:
                saver.restore(sess, checkpoint_path)
                print(f"Model restored successfully from {checkpoint_path}")
            except Exception as e:
                print(f"Error restoring model: {e}")
                return
        else:
            print(f"No checkpoint found in directory: {checkpoint_dir}")
            return

        # --- Prepare Feed Dictionary ---
        feed_dict = {
            model.items: [item_id_to_predict], # Need to be arrays/lists
            model.labels: [0] # Dummy label, not used for prediction score
        }
        # Add ripple set data for the user
        for i in range(args.n_hop):

            feed_dict[model.memories_h[i]] = [user_memories_h[i]]
            feed_dict[model.memories_r[i]] = [user_memories_r[i]]
            feed_dict[model.memories_t[i]] = [user_memories_t[i]]


        scores = sess.run(model.scores, feed_dict=feed_dict)

        # Check the shape and type
        print(f"Scores shape: {scores.shape}, type: {type(scores)}")
        print(f"Scores value: {scores}")

        if isinstance(scores, np.ndarray) and scores.ndim > 0 :
            predicted_score_raw = scores[0]
            predicted_score_prob = 1 / (1 + np.exp(-predicted_score_raw))
            print(f"Raw score: {predicted_score_raw}, Sigmoid prob: {predicted_score_prob}")
            predicted_score = predicted_score_prob
        else:
            # If scores is already scalar, use it directly (and maybe apply sigmoid)
            predicted_score_raw = scores
            predicted_score_prob = 1 / (1 + np.exp(-predicted_score_raw))
            print(f"Raw scalar score: {predicted_score_raw}, Sigmoid prob: {predicted_score_prob}")
            predicted_score = predicted_score_prob

        # Replace the original line:
        # predicted_score = scores_normalized[0]
        # with the logic above to handle potential scalar output.

        print(f"Final Predicted score for user ... : {predicted_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--load_dir', type=str, required=True, help='directory to load model checkpoints from')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--user_id', type=int, required=True, help='User ID (re-indexed) to get recommendations for')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top items to recommend')

    args = parser.parse_args()

    data_info = load_data(args)

    predict(args, data_info, args.load_dir)

    top_items = predict_top_n_for_user(args, data_info, args.load_dir, args.user_id, args.top_n)