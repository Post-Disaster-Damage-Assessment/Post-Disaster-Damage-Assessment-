\# Post-Disaster Building Damage Assessment Using Satellite Imagery



A deep learning project for building-level damage classification using paired pre-disaster and post-disaster satellite imagery from the xBD dataset.



\## Project Overview



Natural disasters can cause extensive damage to buildings. Rapid assessment of building damage can help emergency response teams prioritize affected areas.



This project develops a Siamese deep learning model that compares pre-disaster and post-disaster satellite image crops and classifies each building into one of four damage categories:



\- No Damage

\- Minor Damage

\- Major Damage

\- Destroyed



The project also investigates model performance across disaster events, damage classes, building sizes, and confidence levels.



\## Model Architecture



The classification system uses a Siamese network with a shared-weight ResNet18 backbone.



Both pre-disaster and post-disaster image crops are processed through the same feature extractor. Their feature representations are combined using:



\- Pre-disaster features

\- Post-disaster features

\- Absolute feature difference

\- Element-wise feature product



The combined representation is passed through a fully connected classification head to predict one of four damage classes.



\*\*Input:\*\* Paired RGB building crops, resized to 128 × 128 pixels.



\*\*Output:\*\* Four-class building damage prediction.



\## Dataset



This project uses the xBD (xView2) building damage assessment dataset.



The dataset contains paired satellite imagery and building annotations. Damage labels are obtained from post-disaster annotations.



The dataset is not included in this repository. Users must obtain the dataset separately and arrange it according to the expected project directory structure.



\## Experimental Setup



\- Model: Siamese ResNet18

\- Feature extractor: Shared pretrained ResNet18

\- Input resolution: 128 × 128

\- Number of classes: 4

\- Train-validation split: Scene-level, grouped by disaster event

\- Random seed: 42

\- Validation buildings: 32,151

\- Training buildings: 127,601



The scene-level split prevents buildings from the same scene appearing in both training and validation sets.



\## Experimental Results



The project compares the original Siamese model with a photometric augmentation fine-tuned model.



| Metric | Original | Photometric Augmentation |

|---|---:|---:|

| Accuracy | 87.02% | 87.33% |

| Macro F1 | 0.7442 | 0.7474 |

| Weighted F1 | 0.8738 | 0.8761 |



Photometric augmentation applies synchronized brightness and contrast transformations to paired pre-disaster and post-disaster images.



The experiment produced a modest improvement in overall validation performance. Improvements were not uniform across all disaster events and damage classes.



Detailed metrics, event-wise comparisons, error analysis, calibration results, and building-size experiments are documented in:



\*\*\[Final Experimental Results](FINAL\_EXPERIMENTAL\_RESULTS.md)\*\*



\## Visualizations



The final report includes:



1\. Overall model performance

2\. Original and augmented confusion matrices

3\. Event-wise Macro F1 comparison

4\. Class-wise F1 comparison

5\. Accuracy across building-size groups

6\. Class-wise F1 across building-size groups

7\. Confidence calibration



\## Repository Structure



```text

Post-Disaster-Damage-Assessment/

│

├── building\_dataset.py

├── siamese\_model.py

├── train\_siamese.py

├── train\_siamese\_resume.py

├── finetune\_photo\_aug.py

├── infer\_siamese.py

├── evaluate\_siamese.py

├── evaluate\_validation.py

│

├── analysis scripts

│

├── comparison\_results/

├── validation\_results/

├── building\_size\_comparison/

├── final\_visualizations/

│

├── FINAL\_EXPERIMENTAL\_RESULTS.md

├── requirements.txt

└── README.md

```



\## Installation



Clone the repository:



```bash

git clone https://github.com/Post-Disaster-Damage-Assessment/Post-Disaster-Damage-Assessment-.git

cd Post-Disaster-Damage-Assessment-

```



Create and activate a virtual environment:



```bash

python -m venv venv

```



Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

python
