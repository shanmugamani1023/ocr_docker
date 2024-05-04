FROM python:3.9

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Create directories within the container
RUN mkdir -p /app/gc_pandora \
    && mkdir -p /app/log


# Set the current date in the desired format (YYYY_MM_DD)
RUN CURRENT_DATE=$(date +%Y_%m_%d)

# Copy the contents of the local gc_pandora folder into the container
COPY gc_pandora /app/gc_pandora
COPY strhub /app/strhub


COPY best.pt ocr.ckpt Pretrained.pth tokenizer.pkl ./app/

# Clone the parseq repository
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/baudm/parseq /app/parseq


# Set the working directory
WORKDIR /app/parseq


# Set platform variable
#ENV platform=cpu

# Generate requirements files for specified PyTorch platform
#RUN make torch-${platform}

# Install the project and core + train + test dependencies
RUN pip install -r requirements/train.txt

# Install pip-tools and generate requirements files
RUN pip install pip-tools && \
    cd /app/parseq && \
    make clean-reqs reqs

RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install ultralytics dill pytorch-lightning==1.9.5 timm nltk


WORKDIR /app

COPY . /app

ENTRYPOINT ["python3","app.py"]
