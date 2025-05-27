# RippleNet

This repository is the implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

![](https://github.com/hwwang55/RippleNet/blob/master/framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

A PyTorch re-implementation of RippleNet by Qibin Chen et al. is [here](https://github.com/qibinc/RippleNet-PyTorch).


### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratings.dat`: raw rating file of MovieLens-1M;
  - `tender/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_rehashed.txt`: Мэдлэгийн графын гурвалуудаар задласан файл;
    - `train_data.txt`: Тендерийн өгөгдөл;
- `src/`: implementations of RippleNet.



### Суулгах сангууд
RippleNet нь Python 3.7.0 хувилбартай хослож ажилладаг тул Python 3.7.0-г татаж авч дараах командаар сангуудыг суулгана.

```
pip install -r requirements.txt
```

### Шинээр нэмсэн функцууд
- RippleNet дээр сургасан моделыг хадгалдаг функц байхгүй тул нэмэлтээр хадгалдаг функц нэмсэн. Моделыг .\src\checkpoints дотор хадгалдаг уг зам дээр хадгалдаг.
- Сургалтын үе шатанд л моделын үзүүлсэн үр дүнг мэдэх боломжтой байсныг моделыг хадгалдаг функц нэмж өгснөөр хүссэн үедээ гаргаж ирэх боломжтой болсон.
- Зөвхөн CTR prediction буюу хэрэглэгчийн тухайн зүйл дээр дарах магадлалын acc, auc тооцдог байсан дээр нэмж санал болгох системийн үнэлгээний арга top@k-н precision, recall аргуудыг нэмж өгсөн.

### Оролтын өгөгдлийг боловсруулж сургалтыг хийх код
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book, tender)
$ python main.py --dataset movie
```

### Хадгалсан модел дээр үнэлгээ хийх

```
$ python predict.py --dataset tender --load_dir checkpoints/
```

