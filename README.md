# Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues (ACL 2025 Main)

**Authors:**
**[Youngmin Kim](https://winston1214.github.io)** ,
[Jiwan Chung](https://jiwanchung.github.io) ,
[Jisoo Kim](https://jiiiisoo.github.io/) ,
[Sunghyun Lee](https://romanticbox.github.io) ,
[Sangkyu Lee](https://oddqueue.github.io/) ,
[Junhyeok Kim](https://junhyeok.kim) ,
Cheoljong Yang ,
[Youngjae Yu](https://yj-yu.github.io/home/)

<p align="center">
  <a href="https://arxiv.org/abs/2506.00958">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="#datasets">
    <img src="https://img.shields.io/badge/🤗-Dataset-orange">
  </a>
</p>

<img src='https://github.com/winston1214/nonverbal-conversation/blob/main/imgs/MODEL_PIPELINE.png?raw=true'></img>

### 📊 <a name="datasets"></a> VENUS

- [VENUS-1K](https://huggingface.co/datasets/winston1214/VENUS-1K)🤗: 1,000 annotated samples for prototyping
- [VENUS-5K](https://huggingface.co/datasets/winston1214/VENUS-5K)🤗: 5,000 samples for small-scale training
- [VENUS-10K](https://huggingface.co/datasets/winston1214/VENUS-10K)🤗: 10,000 samples for medium-scale training
- [VENUS-50K](): _Comming soon!_
- [VENUS-100K](): _Comming soon!_ (Full dataset)

#### Load dataset
```python
from datasets import load_dataset
# Change dataset size
train_dataset = load_dataset("winston1214/VENUS-1K", split = "train")
test_dataset = load_dataset("winston1214/VENUS-1K", split = "test")
```


### 📹How to VENUS Collection?

Detailed information is available in the [VENUS folder](https://github.com/winston1214/nonverbal-conversation/tree/main/VENUS).


### Nonverbal-cues Tokenize
<details>
  <summary>Click to expand for more details</summary>

  <b>VQ-VAE Training</b>

  - Item 1
  - Item 2
  - Item 3

</details>



### Acknowlegements
We sincerely thank the open-sourcing of these works where our code is based on:

<a href='https://github.com/yt-dlp/yt-dlp'>yt-dlp</a>, <a href='https://github.com/m-bain/whisperX'>WhisperX</a>, <a href='https://github.com/Junhua-Liao/Light-ASD'>Light-ASD</a>, <a href='https://github.com/IDEA-Research/OSX'>OSX</a>, <a href='https://github.com/radekd91/emoca'>EMOCA</a>, <a href='https://github.com/EricGuo5513/momask-codes'>momask</a>

### Contributions
- **Youngmin Kim**: Dataset Collection and Curation, Model Implementation, Experimentation, Experimental Design, Visualization, Writing, Data Analysis Planning, Project Coordination
- Jiwan Chung: Project Administration, Writing - Review & Editing, Model Pipeline Design
- Jisoo Kim: 3D Visualization, Implementation of 3D Annotation System
- Sunghyun Lee: Data Analysis,  Data Collection Assistance
- Sangkyu Lee: Data Collection Pipeline Design, Model Pipeline Design
- Junhyeok Kim: Mentioned Relevant Papers
- Cheoljong Yang: Funding Acquisition
- Youngjae Yu: Supervision, Corresponding Author


### Misc
Contact [winston1214@yonsei.ac.kr](winston1214@yonsei.ac.kr) for further question.
