
---

# 📘 Deep Learning Experiments: CNNs, nanoGPT, and DistilGPT2

## 1. Project Goals

This project explores multiple deep learning techniques across computer vision and natural language processing. The objectives include:

* Implementing and training **LeNet-5**, a classic convolutional neural network.
* Calculating the total number of **trainable parameters** in the model.
* Experimenting with different training configurations by modifying **batch size**, **learning rate**, and **number of epochs**.
* Training **nanoGPT** on the Shakespeare dataset for character-level text generation.
* Fine-tuning **DistilGPT2** on custom text data and generating text using various decoding strategies.

---

## 2. Summary

This project contains three major components:

1. **Implementation and training of CNNs** using PyTorch on the CIFAR-100 dataset.
2. **Character-level language modeling** using nanoGPT trained on Shakespeare data.
3. **Supervised fine-tuning** of DistilGPT2 on a custom dataset, followed by text generation.

Training may take up to one hour depending on hardware setup.

---

## 3. Environment Setup

This project uses:

* Python 3
* `numpy`
* `torch`
* `torchvision`
* `tqdm`

A Conda environment setup is recommended:

```bash
conda create -n "dl-project" pytorch torchvision torchaudio anaconda::tqdm cpuonly -c pytorch
conda activate dl-project
```

---

## 4. Dataset: CIFAR-100

The CNN models in this project are trained on the **CIFAR-100** dataset:

* 100 classes
* 600 images per class
* Image size: 32×32
* 500 training images + 100 test images per class

Helper scripts are provided to download and prepare the dataset automatically.

---

## 5. Model Implementation

### 5.1 LeNet-5

Implemented using PyTorch with the following layers:

1. Conv2d → ReLU → MaxPool
2. Conv2d → ReLU → MaxPool
3. Flatten
4. Fully connected (256) → ReLU
5. Fully connected (128) → ReLU
6. Fully connected (100)

The forward pass returns:

* model output
* a dictionary containing intermediate feature map shapes from each stage

### 5.2 Parameter Counting

A function computes the total number of **trainable parameters** (in units of millions) using `model.named_parameters()`.

---

## 6. Training Experiments (CNN)

LeNet-5 is trained under multiple configurations, including:

* Default settings
* Batch sizes: 8, 16
* Learning rates: 0.05, 0.01
* Epoch counts: 20, 5

Each configuration produces a trained model and validation accuracy. Results are stored in `results.txt`.

---

## 7. Training nanoGPT on Shakespeare

This project includes a lightweight GPT implementation trained on Shakespeare’s complete works.

### Setup

* Use nanoGPT repository
* Prepare dataset using:

  ```bash
  python data/shakespeare_char/prepare.py
  ```

### Training

A smaller transformer is trained with:

* 4 layers
* 4 attention heads
* Embedding size 128
* Block size 64
* Batch size 12
* 2000 training iterations

Training parameters may be adjusted for experimentation.

### Inference

Generate Shakespeare-like text:

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

Generated samples are saved in `generated_nanogpt.txt`.

---

## 8. Fine-Tuning DistilGPT2 (NLP)

This project fine-tunes **DistilGPT2** on a custom dataset built from WikiText sources.

### Steps:

1. Generate dataset:

   ```bash
   python make_data_csv.py
   ```
2. Train on CPU:

   ```bash
   python distilgpt2_sft_cpu.py --data data.csv --mode train
   ```
3. Implement decoding control and text generation:

   ```bash
   python distilgpt2_sft_cpu.py --mode gen
   ```

Generated text is stored in `distilgpt2.txt`.

---
