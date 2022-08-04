# detectron2 trainer for training on the COCO dataset

데이터 전처리 과정은 1~3 까지의 과정이다.  
각각의 아이디별로 작업된 데이터셋들을 코코데이터셋으로 변경하고 이것을 통합시킨다. 이때 라벨링값들은 아이디값이 슈퍼셋으로 하고 라벨링명의 접두어로 붙여져서 통합된다.  
마지막으로 훈련세트와 검증세트를 나눈다.  
위과정은 훈련머신보다는 수집 머신에서 행해진다.  
그래서 훈연머신에게 전달하는 식으로 처리가 된다.  

## 1. convert coco
pascal voc 포멧을 coco 포멧으로 변환하기  
```sh

python ./coco_tools/voc2coco.py -n frut_nuts --dataset-path ../datasets/fruts_nuts_voc/ --img-path ../datasets/fruts_nuts_voc/voc/ -o ../datasets/fruts_nuts_voc/anno.json

python ./coco_tools/voc2coco.py -n dic_1004 --dataset-path ../../datasets/cauda_test/dic_1004/ --img-path ../../datasets/cauda_test/dic_1004/voc/ -o ../../datasets/cauda_test/dic_1004/anno.json

```
## 2. merge coco files

```sh
python ./coco_tools/coco_merge.py -c ./settings/cmd.yaml
```
## 3. split coco files

```sh
python ./coco_tools/coco_spliter.py  --json-path=../datasets/fruts_nuts_voc/anno.json --output-path=../datasets/fruts_nuts_voc/ --img-path=../datasets/fruts_nuts_voc/  --train-ratio=0.8

```
## 4. config 파일 만들기 


```sh
python make_cfg.py -s ./settings/cauda.yaml
``` 
## train

```sh
CUDA_VISIBLE_DEVICES=0 python train_net.py

python train_net.py --config-file ./configs/frut_nuts_config.yaml -s ./settings/sample.yaml --num-gpus 1
python train_net.py --config-file ./configs/micro_controller_config.yaml -s ./settings/micro_controller.yaml --num-gpus 1
```
## 탠서보드 

--logdir 옵션으로 학습결과물이 출력되는 디랙토리를 지정한다.<br>

```sh
tensorboard --logdir /home/ubiqos-ai2/work/visionApp/cauda_project/output/all
```
http://localhost:6006 으로 접속한다.<br>
