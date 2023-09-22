![Wristbone_Fracture_Detection (10)-1](https://github.com/ACM-Research/CarpusFractureDiagnosticsYOLOv7-Public/assets/78242653/9a13ffc1-ea4d-48d9-8859-4ff4a1942bbd)

# Introduction

X-ray image interpretation requires time-consuming specialized training from professionals who’ve
been in the field for decades. Moreover, Pediatric Surgeons in training or emergency physicians
often interpret radiographs wrong due to inexperience or time constraints. In many countries,
shortages of radiologists were reported, posing a risk to patient care. With this being said, distal
radius and ulna fractures account for the majority of pediatric wrist injuries with an incidence peak
in adolescence. Hence, due to astonishing progress in computer vision algorithms, automated
fracture detection has become a topic of research interest. Several uses of artificial intelligence in
object detection in recent years have shown promising results in the diagnosis of such fractures.
Using Yolov7, we have created a model that helps detect wrist fractures and other injuries with
accuracy. Our model will aid radiologists in making more efficient and accurate diagnoses of wrist
injuries.

# Application

Our machine learning model aims to aid or replace surgeons in areas with a shortage, particularly in
orthopedic surgery. Even in areas where there are sufficient surgeons, misdiagnosis is a significant
issue, with up to 26.2% of patients with perilunate dislocation not diagnosed correctly. If left
untreated, health complications such as arthritis, deformity, and instability invariably develop within
5 years, which can lead to significant disability. The associated cost and morbidity implications
furthermore present significant harms. Our model can serve as a rapid and reliable reporting
system to improve diagnosis rates, particularly for rare conditions like perilunate dislocation, and
other carpal fracture detection where surgeons may have less experience and may be more prone
to making errors. This also could make substantial improvement in night-time diagnoses, where
misdiagnoses are statistically more likely to occur, due to possible fatigue from professionals. It
can serve as a supplementary aid to help surgeons detect issues they may have missed.
# Dataset


![image](https://github.com/ACM-Research/CarpusFractureDiagnosticsYOLOv7-Public/assets/78242653/12cab38f-c4c3-49c3-9877-72f7b8a2d5a0)

Figure 1. Sample images of dataset

The dataset we used is from the public and free to use GRAZPEDWRI-DX dataset that was
produced at the Department for Pediatric Surgery of the University Hospital Graz for the purpose
of research and innovation to help benefit the future of healthcare. All the images have been
collected over the course of 10 years (2008-2018), and all the patients have been de-identified to
maintain their privacy.

- [x] Radiographs of 6,091 patients
- [x] Ages ranging from 0.2 to 19 years, with a mean age of 10.9 years
- [x] From 10,643 studies (20,327 images)
- [x] 2,688 females, 3,402 males, 1 unknown


# YOLOv7/v8

YOLO(You Only Look Once) is an efficient and accurate object detection algorithm that uses deep
convolutional neural networks to recognize and localize objects in an image. Unlike traditional
object detection algorithms that require multiple passes through an image, YOLO divides the
image into a grid and predicts the bounding boxes and class probabilities for each grid cell in a
single pass. YOLOv7 is one of the latest releases of YOLO, around 120% faster than previous
iterations held at the same accuracy, using improved training techniques. This algorithm could
be used in medical applications to speed up processes, saving time and reducing load, as well as
provide potentially more accurate diagnoses. YOLOv8 was released very recently, with 5 versions being
available of different sizes, we use the smallest model ’yolov8n’ due to larger ones requiring
greater computational power + longer training times.
# Analysis
## Model Training

• The pre-trained model was trained off of the same dataset that we are using, hence explaining
the relatively similar accuracy performance with the trained model

• The YOLOv7 and YOLOv8 trained models were split into a train/test/valid dataset group, with
14249 in the train dataset - 70% | 2000 in the test dataset - 10% | 4078 in the validation dataset - 20%

• Models were trained on the training dataset and weighed on the test, and we ran inferences
on these models on the validation dataset of N = 4078. Standard hyperparameters were used
on training, except for 50 epochs due to the lack of prolonged computational resources
necessary for greater epochs. Was trained on the Google Colab Pro platform, with Premium
GPU classes (typically being NVIDIA V100 and A100 Tensor Core) and High-RAM class of 32
GB

## Model Evaluation
![image](https://github.com/ACM-Research/CarpusFractureDiagnosticsYOLOv7-Public/assets/78242653/cd0d832b-fd1f-4eb1-ae56-6f361e091c48)

Figure 2. Model evaluation result

In object detection, precision, recall, and mean average precision (mAP) are commonly used
metrics to evaluate the performance of an algorithm.

• Precision measures the percentage of correctly predicted positive samples out of all predicted
positive samples, where a high precision means most detections made are correct.

P recision =
tp
tp + fp
(1)

• Recall measures the percentage of correctly predicted positive samples out of all true positive
samples, where a high recall means most objects can be detected.

Recall =
tp
tp + fn
(2)

• Mean Average Precision (mAP) is a popular metric used to evaluate the overall performance of
an object detection algorithm. It is calculated by averaging the average precision (AP) scores for
each class over a range of IoU (Intersection over Union) thresholds, usually held standard at 0.5.

• The mAP at an IoU threshold of 0.5 (mAP@0.5) is commonly used to evaluate the
performance of object detection algorithms, as it is a good measure of the algorithm’s ability to
detect objects with high precision and recall.

# Conclusion

YOLOv7/v8 is a powerful tool for detecting fractures in X-rays with high accuracy and speed. It
can help automate the diagnosis of fractures and assist radiologists in making faster and more
accurate diagnoses. The use of YOLOv7/v8 in clinical settings has the potential to improve patient
outcomes and reduce healthcare costs.

![image](https://github.com/ACM-Research/CarpusFractureDiagnosticsYOLOv7-Public/assets/78242653/a6c00caf-17bb-4657-8f86-822974467c84)

Figure 3. Confusion Matrix, Precision-Recall Curve, F1 Score Curve

• Confusion Matrix We achieved a remarkable prediction accuracy of 0.90 for fracture, where
90% of fractures were identified, the rest unclassified. Other metrics and features labeled by the
radiologists were also shown with varying degrees of accuracy.

• Precision-Recall Curve The precision-recall curve shows the tradeoff between precision and
recall for different thresholds. We see that the model has high precision and recall for detecting
fracture, text, and metal.

• F1 Score Curve F1 score is helpful in determining the optimum confidence that balances the
precision and recall values for the model with the overall model performance from 0(worst) to
1(best). For fracture, the confidence value that optimizes the precision and recall is around 0.5.

• Output of the model

![image](https://github.com/ACM-Research/CarpusFractureDiagnosticsYOLOv7-Public/assets/78242653/100a2f84-5285-409e-aa05-faba405f669a)

Figure 4. Labeled Dataset and Predictions

# References

Tingle, C. (2022, May 25). Workforce shortage impacts all areas of orthopedics. Healio. Retrieved
April 23, 2023, from https://www.healio.com/news/orthopedics/20220523/workforce-shortageimpacts-all-areas-of-orthopedics
Colak, I. et al. (2017, November 29). Lack of experience is a significant factor in the missed
diagnosis of perilunate fracture dislocation or isolated dislocation. Acta Orthopaedica et Traumatologica Turcica. Retrieved April 23, 2023, from https://www.sciencedirect.com/science/article/pii/S1017995X16300773abs0010
Nagy, E., Janisch, M., Hržić, F. et al. A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for
machine learning. Sci Data 9, 222 (2022). https://doi.org/10.1038/s41597-022-01328-z
Hayat Z, Varacallo M. Scaphoid Wrist Fracture. 2023 Jan 30. In: StatPearls [Internet]. Treasure
Island (FL): StatPearls Publishing; 2023 Jan–. PMID: 30725592.
