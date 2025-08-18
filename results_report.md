# Multi-person detection for identity verification - Report of results

## 1. Considerations

Due to the small size of the dataset (17 videos), it was extremely difficult to train a model from scratch.  
If the data were split into training, validation, and test sets, each subset would be too small to properly evaluate generalization or detect bias.

For this reason, I designed approaches based on pre-trained models, focusing primarily on engineering and pipeline design rather than model training.

The development followed an iterative process: starting with the simplest approach and refining it based on the errors observed in the previous iteration.The following sections describe each approach and the results obtained.


---

## 2. System and approaches

The system follows a modular pipeline:

```
video → frame extraction → person detection → aggregation of frame-results → video label (0/1)
```

1. **Frame extraction:** Sample frames from the video (configurable rate).  
2. **Person detection:** Detect people in each frame using the selected backend.  
3. **Aggregation strategy:** Convert frame-level detections into a single video-level label.  
4. **Output:** Binary label indicating whether multiple people are present.

All approaches follow this pipeline, differing mainly in the aggregation strategy (with the exception of the last strategy, which adds an additional model).

Four approaches have been designed: `counting`, `temporal`, `temporal_cardaware`, and `temporal_textaware`.
Quantitative results are extracted from the reports that are automatically generated when an evaluation is runned.

### 2.1 Counting Strategy

This strategy is the most basic of those implemented. Each frame is processed by the detection model to count the number of persons detected.
To aggregate the results, we compute the proportion of frames where more than one person is detected.  
If this proportion exceeds the `people-threshold` (threshold selected by the user), the video is labeled as containing multiple persons.

For configuration, I used a `people-threshold` of 0.0, which means that a video is classified as `1` (multiple persons) if any frame contains more than one person detected.

To reproduce the evaluation:

``
python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution counting --people-threshold 0.0 --verbose
``

The results are:

#### Performance metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.706 |
| Precision | 0.625 |
| Recall | 0.714 |
| F1-Score | 0.667 |

As we can see, the results are relatively low across all metrics, indicating that this method does not perform well. Examining the confusion matrix:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 7 (TN) | 3 (FP) |
| **Actual 1** | 2 (FN) | 5 (TP) |

We observe 3 false positives and 2 false negatives.

* False positive videos (predicted 1, true 0): veriff3, veriff8, veriff18.

    * veriff3 clearly contains only one person, making this an unexpected false positive.
    * veriff8 and veriff18 are more complex cases because they contain photos in ID cards, which the model likely detects as a person.

* False negative videos (predicted 0, true 1): veriff2, veriff19.
    * Both videos clearly contain two persons, so the model fails to detect them.

To address this, the next step is to reduce the detection confidence threshold (0.5 by default) to ensure all persons are recognized. This will increase the number of false positives, but we will handle those in the next approach. The priority here is for the model to detect all actual persons first.

So, as second experiment, I am evaluating with a detector threshold of 0.25:
`` python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution counting --people-threshold 0.0 --threshold 0.25``

In this case, the results are as follows:

| Metric | Value |
|--------|-------|
| Accuracy | 0.588 |
| Precision | 0.500 |
| Recall | 1.000 |
| F1-Score | 0.667 |

And the confusion matrix:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 3 (TN) | 7 (FP) |
| **Actual 1** | 0 (FN) | 7 (TP) |

As we can see, there are no false negatives, meaning the model now successfully detects all persons in the frames.
The issue, however, is the large number of false positives (7): `veriff1`, `veriff3`, `veriff8`, `veriff11`, `veriff14`, `veriff17`, and `veriff18`.

These false positives are mainly due to:
- Photos in ID cards being detected as persons.
- Artifacts in the frame, which are more likely to be detected as persons because the detection confidence threshold was lowered to 0.25.

Focusing first on the artifacts detected as persons (the photos on the ID cards will be addressed later, since we first need the model to correctly detect the "basic" samples), I have designed the next strategy, which takes temporal information into account.


### 2.2 Temporal strategy (hysteresis)

The issue I want to address now is the detection of artifacts as persons, which cause false positives from the model. To handle this, I have designed a strategy that applies temporal hysteresis in the following way:

If multiple persons are present in a frame, it is likely that they will remain visible for several consecutive frames. With this strategy, we do not classify a video as a multi-person video unless some persons are detected for at least X consecutive frames. The number of frames is set by the user using the `temporal-min-consecutive` parameters.

In this way, if the detector produces a false detection due to an artifact, camera movement, or similar causes, this strategy will filter out those false positives.

So, I runned an evaluation with following command:

```python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution temporal --threshold 0.25 --temporal-min-consecutive 5```

The results are:

#### Performance metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.588 |
| Precision | 0.500 |
| Recall | 0.714 |
| F1-Score | 0.588 |

And the confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 5 (TN) | 5 (FP) |
| **Actual 1** | 2 (FN) | 5 (TP) |

As we can see, the results are similar. The two false negative videos are `veriff2` and `veriff9`, which were the reason for lowering the confidence threshold significantly. These videos seem to be particularly challenging for this model (``YOLOv8n``), as it only detects multiple persons in a few frames and with low confidence.

Regarding the false positives, we have reduced the failures to five cases: `veriff1`, `veriff8`, `veriff11`, `veriff17`, and `veriff18`. All these videos contain ID card photos, which are probably being detected as persons.

Therefore, the next strategy will focus on filtering out detections corresponding to ID cards.


### 2.3 Card-aware strategy

The temporal strategy reduced false positives caused by artifacts but still struggled with ID card photos being detected as persons. To address this specific issue, I implemented a card-aware filtering approach.

The strategy combines temporal hysteresis (previous strategy) with geometry-based filtering to identify and filter out faces that are likely part of ID cards or documents:

- First, the largest person detected is considered the real person, and its size is taken as reference.
- After this, each frame with more than one person detected is filtered with:

1. **Area comparison**: Small detections are considered ID card candidates if their area is less than ``card-min-area-ratio``% of the largest detection (the reference).
2. **Shape analysis**: Detections with aspect ratios close to 1.0 (square-like) are more likely to be ID card photos, within a tolerance of ``card-square-tolerance``.

If a detection meets **both conditions** (is smaller *and* has an aspect ratio close to 1), that detection is considered an ID card face and is rejected.
After applying the filter, the temporal consistency from the previous strategy (parameter ``temporal-min-consecutive``) can also be enforced.


To reproduce the evaluation:

```
 python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution temporal_cardaware --threshold 0.25 --temporal-min-consecutive 4 --card-min-area-ratio 0.7 --card-square-tolerance 0.85 --verbose
```

#### Performance metrics

The results are:

| Metric | Value |
|--------|-------|
| Accuracy | 0.706 |
| Precision | 0.625 |
| Recall | 0.714 |
| F1-Score | 0.667 |

And the confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 7 (TN) | 3 (FP) |
| **Actual 1** | 2 (FN) | 5 (TP) |

As we can see, this approach reduced two false positives, indicating that the strategy works as expected for those two videos. However, there are still failures, and the overall metrics remain relatively low (F1-score of 0.667).

Videos with remaining false positives: `veriff1`, `veriff8`, and `veriff17`.

Given the small size of the dataset, it is difficult to fine-tune parameters and ensure generalization. To address this, I designed an additional strategy specifically focused on detecting ID card photos.

### 2.4 Text-aware strategy (OCR)

The card-aware strategy used ID card filtering through geometric constraints, but it still had clear limitations: many ID cards do not match the expected geometric patterns, and the method requires careful parameter tuning. To overcome this, I designed a more robust approach based on **text detection**.

The idea is: ID cards and documents usually contain text, while real faces in videos almost never have text nearby. Leveraging this observation, I combined OCR with temporal hysteresis to filter out detections that are likely document photos rather than real people.

The process works as follows: for frames with at least two detections, OCR (implemented with EasyOCR) is applied to identify text regions. Each detected face is then compared with the location of text, and if it falls within a given proximity threshold, it is discarded as a likely ID card face. After this frame-level filtering, the usual temporal hysteresis is applied to ensure stability over time.

The method is controlled by two parameters:
- `text-confidence-threshold`: minimum OCR confidence required to consider a text region (default: 0.5).
- `text-proximity-threshold`: maximum distance in pixels between a face and text to be filtered (default: 100).


To reproduce the evaluation:

```
python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution temporal_textaware --threshold 0.25 --temporal-min-consecutive 2 --text-confidence-threshold 0.3 --text-proximity-threshold 100 --sample-rate 5 --verbose
```
#### Performance metrics

The results are:

| Metric | Value |
|--------|-------|
| Accuracy | 0.706 |
| Precision | 0.625 |
| Recall | 0.714 |
| F1-Score | 0.667 |

And the confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 7 (TN) | 3 (FP) |
| **Actual 1** | 2 (FN) | 5 (TP) |

As we can see, we get the exact same results. Due to my capacity limitations (lack of GPU), I have had to use a `sample-rate` of 5, so I am probably losing some information.

Without being able to tune the parameters because of the GPU limitation, I have not been able to detect all ID cards using the last two strategies.


### 2.5 Optimization

While I was reviewing the videos, I noticed that in most of them, the ID card is shown only for a short time (as in a verification). Since I have not been able to tune the parameters of my solution, I believe that I can improve the results using the most basic strategy (`counting`), taking advantage of the fact that the ID cards appear for a short duration in the videos.

I have made the next evaluation:

```python -m src.main evaluate --dataset-path "data/raw/Data Scientist Take-home Assignment - Dataset" --labels-file "data/raw/Data Scientist Take-home Assignment - Dataset/labels.txt" --solution counting --people-threshold 0.2 --threshold 0.25```

#### Performance metrics

The results are:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7647 |
| Precision | 0.714 |
| Recall | 0.714 |
| F1-Score | 0.714 |

The confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---------|-------------|-------------|
| **Actual 0** | 8 (TN) | 2 (FP) |
| **Actual 1** | 2 (FN) | 5 (TP) |

As we can see, we achieved our best results with the most "basic" strategy. While I could probably optimize this further with more tuning, I think it doesn’t make sense because it is not generalizable (or at least, we cannot know without more data). In these videos, the time an ID appears may be short, but in others it could be longer, and we don’t have enough data to verify this.

---



## 3. Improvements with more time and data

One major limitation has been my resources. Without a GPU, my processing is slow, and I have not been able to tune the parameters (for example, I added a few model backends, but I only had time to test YOLO). Also, with more time, I would have had the opportunity to design more robust solutions. Some ideas that I had but could not implement due to time and lack of data are:

1. **Parameter optimization and model selection:**
   - With a larger dataset, systematic tuning of thresholds (detection confidence, temporal hysteresis, card-area ratios, OCR proximity) could improve precision and recall while ensuring generalization.
   - Evaluate multiple pre-trained detection backends (YOLOv8, Faster R-CNN, SSDLite, etc.) and select or ensemble the best-performing model(s).

2. **Training or fine-tuning models:**  
   - With sufficient labeled data, train a dedicated multi-person detection model tailored for ID verification scenarios.
   - Use temporal models (e.g., LSTM, Temporal convolutional networks...) to capture motion patterns over consecutive frames, helping differentiate between real people and artifacts like ID cards.

3. **Advanced ID card filtering:**  
   - Use specialized ID card detection models to reliably identify document faces regardless of geometry or orientation.
   - Employ semantic analysis (e.g., CLIP or similar vision-language models) to differentiate actual human faces from printed images or cards.

4. **Multi-modal analysis:**  
   - Integrate depth information to verify that detected objects are 3D people rather than flat images.

5. **Optimized frame sampling and inference:**  
   - With GPU resources, higher frame sampling rates can be used to improve temporal consistency.
   - Combine frame-level predictions with video-level aggregation using learnable or probabilistic approaches, instead of fixed thresholds.

6. **Data augmentation and synthetic data:**
   - Generate synthetic variations of videos (different lighting, backgrounds, orientations) to increase model robustness.
   - Also, we can use simulated ID cards in videos to improve the system’s ability to distinguish real faces from artifacts.

7. **Fair evaluation:**  
   - With a larger dataset, applyhold-out test sets to ensure unbiased evaluation of model performance.
