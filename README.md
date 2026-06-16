# Facial Expression Recognition Challenge

-------------------------------------------------------------------------------------------------
რეპორტის ლინკი: 
https://wandb.ai/gormo22-free-university-of-tbilisi-/fer-facial-expression/reports/Facial-Expression-Recognition-Challenge--VmlldzoxNzI0NjgzNg?accessToken=xhje98grgn380oz8ffpzy0w5k9l91wxs2v1779v8p9ps8z7pxyvhxmj1c9q6ughy
-------------------------------------------------------------------------------------------------

> Kaggle FER2013 Challenge | PyTorch | WandB | Google Colab

---

## პროექტის მიმოხილვა

ამ პროექტში გადავჭერი Kaggle-ის **Facial Expression Recognition Challenge** - სახის სურათებით 7 ემოციის კლასიფიკაცია (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). მოდელები PyTorch-ში დავწერე, დავატრენინგე Google Colab-ის T4 GPU-ზე, ექსპერიმენტები WandB-ზე დავლოგე, საბოლოოდ კი ავაწყვე რეპორტი, სადაც ლოგებისა და პლოტების დახმარებით პროექტი მიმოვიხილე.

---

## მონაცემები (Dataset)

**წყარო:** [Kaggle FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

- **ფორმატი:** CSV (`train.csv`, `test.csv`) — პიქსელები space-separated string-ად
- **სურათები:** 48×48, გრეისქეილი (1 channel)
- **კლასები:** 7 ემოცია
- **სატრენინგო ნიმუშები:** ~28,709
- **ვალიდაცია:** 10% stratified split (`random_state=42`)
- **კლასების დისბალანსი:** Happy კლასი მნიშვნელოვნად ჭარბობს; Disgust — ყველაზე ნაკლები რაოდენობით გვხვდება. 

### Dataset კლასი

მონაცემები შეიცავს 48×48 პიქსელიან გრეისქეილ სახის სურათებს, რომლებიც CSV-ში პიქსელების სტრიქონებად ინახება. სატრენინგო სეტი ~28,000 სურათია. ვალიდაციისთვის გამოვყავით 10% სტრატიფიცირებული split-ით.
გამოყენებული ტრანსფორმაციები (train):
RandomHorizontalFlip - სახე სიმეტრიულია, ამიტომ ბრუნვა ლეგიტიმური augmentation-ია
RandomRotation(10°) - მცირე ბრუნვა ვარიაციის გაზრდისთვის
Normalize(mean=0.5, std=0.5) - პიქსელები [-1, 1] დიაპაზონში

ვალიდაციასა და ტესტზე მხოლოდ ToTensor + Normalize გამოვიყენე augmentation-ის გარეშე.

---

## Data Augmentation

| ტრანსფორმაცია | Train | Val/Test | მიზანი |
|---|---|---|---|
| `RandomHorizontalFlip` | კი | არა | სახის სიმეტრიულობის გამოყენება |
| `RandomRotation(10°)` | კი | არა | პოზის ვარიაცია |
| `ToTensor` | კი | კი | პიქსელები [0,1]-ში |
| `Normalize(0.5, 0.5)` | კი | კი | [-1, 1] range |

> **შენიშვნა:** Architecture 2-ში augmentation-ი განზრახ გამოვრთე, რათა მენახა, წაიყვანდა თუ არა მოდელს ოვერფიტისკენ.

**Batch size:** 64 | **num_workers:** 2

---

## Training Loop და Utilities

- **Loss:** `CrossEntropyLoss`
- **Optimizer:** `Adam` (weight_decay კონფიგით)
- **Scheduler:** `StepLR(step_size=10, gamma=0.5)` — lr ყოველ 10 epoch-ში ×0.5
- **WandB logging:** train/val loss, train/val acc, learning rate

---

## არქიტექტურები

### არქ. 1 - TinyCNN (Baseline)

```
Conv(1→16) → ReLU → MaxPool(2)
Conv(16→32) → ReLU → MaxPool(2)
Flatten → Linear(32·12·12 → 128) → ReLU → Linear(128 → 7)
```
რა გამოვიყენე: 2 კონვოლუციური ლეიერი (16→32 ფილტრი), MaxPool, 2 სრულად დაკავშირებული ლეიერი.
რატომ: baseline-ის დასამყარებლად - მინდოდა მენახა, რა შედეგს გვაძლევს ყველაზე მარტივი CNN augmentation-ისა და რეგულარიზაციის გარეშე.
კონფიგი: lr=1e-3, epochs=20, weight_decay=0
მოსალოდნელი: ~40-45% val accuracy - დაბალი, მაგრამ გასაგები მარტივი მოდელისთვის.
მიღებული: მოდელი სწრაფად დაოვერფიტდა, train accuracy-მ 63%-ს მიაღწია, val კი 55-56%ის ირგვლივ გაჩერდა მეოცე ეპოქაზე, ტრენინგი რომ გაგრძელებულიყო gap ცალსახად გაიზრდებოდა.
**დასკვნა:** augmentation და regularization საჭიროა.

---

### არქ. 2 - MediumCNN (BatchNorm + Dropout, **augmentation გარეშე**)

```
[Conv(1→32) → BN → ReLU → MaxPool] ×3  (32→64→128)
Flatten → Linear(128·6·6 → 256) → ReLU → Dropout(0.5) → Linear(256 → 7)
```
რა გამოვიყენეთ: 3 კონვოლუციური ბლოკი (32→64→128), BatchNorm2d, MaxPool, Dropout(0.5) classifier-ში.
განსხვავება: ამჯერად მოდელი augmentation-ის გარეშე დავატრენინგე (run_experiment_no_aug), რათა მენახა BatchNorm/Dropout-ის ეფექტი ცალ-ცალკე.
კონფიგი: lr=1e-3, epochs=20
მოსალოდნელი: ოვერფიტი გაიზრდება - augmentation-ის გარეშე.
მიღებული: train accuracy 73%+, val ~55%. ოვერფიტი კვლავ არის. 
გვიჩვენა, რომ data augmentation კრიტიკულია.

**მიზანი:** BatchNorm/Dropout-ის იზოლირებული ეფექტის ჩვენება augmentation-ის გარეშე  
**კონფიგი:** `lr=1e-3`, `epochs=20`  

---

### არქ. 3a - DeepCNN (Double-Conv + Dropout2d + L2)

```
[Conv→BN→ReLU, Conv→BN→ReLU → MaxPool → Dropout2d(0.25)] ×3  (32→64→128)
Flatten → Linear(→512) → ReLU → Dropout(0.5)
         → Linear(→256) → ReLU → Dropout(0.3) → Linear(→7)
```
რა გამოვიყენე: 3 ბლოკი, თითოში 2 Conv ლეიერი; Dropout2d(0.25) ყოველ ბლოკში; classifier-ში Dropout(0.5) + Dropout(0.3); L2 weight_decay=1e-4.
ახლა augmentation ჩართულია (RandomHorizontalFlip + RandomRotation(10°)).
კონფიგი: lr=5e-4, epochs=25, weight_decay=1e-4
მოსალოდნელი: val accuracy 50%+ ოვერფიტის შემცირებასთან ერთად.
მიღებული: val accuracy გაიზარდა, train/val gap შემცირდა. ეს პირველი მოდელი იყო, სადაც augmentation + regularization სინერგია ნათლად გამოჩნდა.

**ახლა augmentation ჩართულია**  
**კონფიგი:** `lr=5e-4`, `epochs=25`, `weight_decay=1e-4`  
**შედეგი:** val acc ~53% | train/val gap მნიშვნელოვნად შემცირდა  
**Key insight:** Dropout2d (channel-level dropout) + L2 + augmentation = სინერგია

---

### არქ. 3b - BetterDeepCNN (GELU + Global Average Pooling)

```
[Conv→BN→GELU, Conv→BN→GELU → MaxPool → Dropout2d(p)] ×4  (32→64→128→256)
AdaptiveAvgPool2d(1) → Flatten → Linear(256→256) → GELU → Dropout(0.5) → Linear(→7)
```
რადგან ამ მიდგომამ gap შეამცირა, ეპოქების რაოდენობის გაზრდასთან ერთად დავამატე ტექნიკური ცვლილებები.
რა გამოვიყენე: 4 კონვოლუციური ბლოკი (32→64→128→256), ReLU-ის ნაცვლად GELU activation, AdaptiveAvgPool2d(1) (Global Average Pooling) Flatten-ის ნაცვლად, გაზრდილი Dropout (0.25→0.3→0.4 ბლოკებზე).
ძირითადი ტექნიკური ცვლილებები:
GELU - smooth activation, Transformer-ებში პოპულარული; ReLU-ზე ხშირად უკეთეს გრადიენტს იძლევა
Global Average Pooling - feature map-ების სრულ Flatten-ს ანაცვლებს, პარამეტრებს ამცირებს, generalization-ს აუმჯობესებს
კონფიგი: lr=3e-4, epochs=60, weight_decay=5e-4
მიღებული: train accuracy - 66%, val accuracy ~65%, ბევრად სტაბილური ტრენინგი. (60 ეპოქა)


**ძირითადი ცვლილებები:**

| ელემენტი | ძველი | ახალი | მიზეზი |
|---|---|---|---|
| Activation | ReLU | GELU | სიმულუვრი gradient flow |
| Pooling (classifier) | Flatten | AdaptiveAvgPool2d(1) | ნაკლები პარამეტრი, უკეთესი generalization |
| Depth | 3 ბლოკი | 4 ბლოკი | უფრო ღრმა feature hierarchy |
| Dropout | 0.25 everywhere | 0.25→0.3→0.4 | progressive regularization |

---

### არქ. 4 - CNN + BiLSTM Hybrid

```
CNN Feature Extractor:
  Conv(1→64) → BN → ReLU → MaxPool
  Conv(64→128) → BN → ReLU → MaxPool → Dropout2d(0.25)

Reshape: (B, C, H, W) → (B, H, C·W)   # H rows = temporal sequence

BiLSTM:
  input_size = 128·12 = 1536
  hidden = 128, layers = 3, bidirectional = True
  output: last hidden state → size 256

Classifier:
  Dropout(0.3) → Linear(256→128) → ReLU → Linear(128→7)
```

რა გამოვიყენე: CNN ამოიღებს spatial feature-ებს, feature map-ი გარდაიქმნება sequence-ად და გადაეცემა BiLSTM-ს.
Feature Map-დან:
CNN output: (B, C, H, W)
→ permute: (B, H, C, W)
→ reshape:  (B, H, C*W)   ← H სტრიქონი გახდა "დროის ნაბიჯები"
→ BiLSTM → ბოლო hidden state → classifier
კონფიგი: lr=5e-4, epochs=30, weight_decay=1e-4
მოსალოდნელი: LSTM-ს შეუძლია inter-row კავშირების დაჭერა.
მიღებული: train accuracy - 61%, val accuracy 60%. 30-ე ეპოქისთვის BetterDeepCNN-ზე ოდნავ ნაკლები.

**ჰიპოთეზა:** სახის row-by-row წაკითხვა BiLSTM-ით ჰორიზონტალურ კონტექსტს დაიჭერს (თვალები, პირის გამომეტყველება)

**ტესტირებული hyperparameter-ები:**

| პარამეტრი | ვარიანტი 1 | ვარიანტი 2 (საბოლოო) |
|---|---|---|
| cnn_channels | (32, 64) | (64, 128) |
| lstm_layers | 2 | 3 |
| lstm_hidden | 128 | 128 |
| bidirectional | True | True |
| dropout | 0.3 | 0.3 |

**Trainable params:** ~3.2M  
მიღებული: train accuracy - 61%, val accuracy 60%. 30-ე ეპოქისთვის BetterDeepCNN-ზე ოდნავ ნაკლები. ტრენინგი სტაბილური.

---

## შედეგების შეჯამება

| მოდელი | Val Acc | Epochs | ძირითადი Feature |
|---|---|---|---|
| TinyCNN | ~56% | 20 | baseline, no reg |
| MediumCNN (no aug) | ~55% | 40 | BN + Dropout |
| DeepCNN | ~60% | 30 | Dropout2d + L2 + aug |
| BetterDeepCNN | ~65% | 60 | GELU + GAP + 4 blocks |
| CNN+BiLSTM | ~60% | 30 | sequential processing |

---

## WandB პროექტი

პროექტი: `fer-facial-expression`

| Run Name | არქიტექტურა |
|---|---|
| `arch1-tiny-cnn-baseline` | TinyCNN |
| `arch2-medium-cnn-bn-overfit-demo` | MediumCNN |
| `arch3-deep-cnn-aug-regularized` | DeepCNN |
| `better-deepcnn-gelu-gap` | BetterDeepCNN |
| `arch4-cnn-lstm-hybrid` | CNN+BiLSTM |


## ანალიზი:

TinyCNN — მოდელი ძალიან მარტივია ამოცანისთვის. gap მიუთითებს, რომ ქსელს feature-ების სრულად დასამახსოვრებლად სირთულე არ ყოფნის, მაგრამ გენერალიზაცია მაინც სუსტია — underfit და overfit ერთდროულად.
MediumCNN (no aug) — ყველაზე პრობლემური შემთხვევა. 18%-იანი gap ძლიერი ოვერფიტია: მოდელმა სატრენინგო სეტი "დაიზეპირა", augmentation-ის გარეშე კი ახალ მონაცემებზე ვერ გენერალიზდება. BatchNorm და Dropout-ი საკმარისი არ აღმოჩნდა data augmentation-ის გარეშე.
DeepCNN — augmentation-ისა და L2 რეგულარიზაციის დამატებამ gap 18%-დან ~8%-მდე დაიყვანა. მნიშვნელოვანი გაუმჯობესება, მაგრამ ოვერფიტი კვლავ შესამჩნევია — Dropout2d + weight_decay კომბინაცია ჯერ არ არის საკმარისი.
BetterDeepCNN — ყველაზე წარმატებული. 1%-იანი gap ნიშნავს, რომ GELU activation, Global Average Pooling და პროგრესული Dropout (0.25→0.4) ეფექტურად აგვარებს რეგულარიზაციის პრობლემას. 60 ეპოქის მიუხედავად მოდელი არ დაოვერფიტდა — GAP-ი განსაკუთრებით გადამწყვეტი იყო, რადგან flat-ening-ის ნაცვლად channel-wise averaging ბევრ ზედმეტ პარამეტრს შლის.
CNN+BiLSTM — 1%-იანი gap BetterDeepCNN-ის მსგავსია, მაგრამ აბსოლუტური val acc (60%) ოდნავ დაბალია. LSTM-ი row-by-row ინფორმაციას სწავლობს, რაც განსხვავებულ inductive bias-ს ქმნის — ოვერფიტი კარგადაა კონტროლირებული, მაგრამ pure CNN-ის spatial feature extraction-ს ვერ სჯობს ამ ამოცანაზე.

ზოგადი დასკვნა: ყველაზე მეტი გაუმჯობესება მოვიდა არა მოდელის სიღრმის გაზრდით, არამედ data augmentation-ის დამატებით (MediumCNN→DeepCNN: gap 18%→8%) და GAP + პროგრესული Dropout-ით (DeepCNN→BetterDeepCNN: gap 8%→1%). ეს გვიჩვენებს, რომ FER2013-ზე რეგულარიზაციის სტრატეგია მნიშვნელოვანია.
