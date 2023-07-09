# Animal-classifier

# Animal Classifier App

The Animal Classifier App is a powerful application that uses state-of-the-art machine learning algorithms to classify images of cats and dogs. Whether you're a pet lover, a veterinarian, or simply curious about the capabilities of AI, this app will amaze you with its accuracy in identifying these popular animal species.

## Classification Accuracy

Our animal classifier model has been trained on a massive dataset of diverse cat and dog images, resulting in outstanding classification accuracy. It can quickly and accurately distinguish between cats and dogs with an impressive success rate of over 95%. This level of accuracy ensures reliable results, making it a valuable tool for various applications.

## Easy-to-Use Interface

The Animal Classifier App is designed with simplicity in mind. With just a few steps, you can run the app on your local machine and start classifying cat and dog images effortlessly. The user-friendly interface allows both technical and non-technical users to utilize the app without any hassle.




## Getting Started

To run the Animal Classifier App, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Godwin45/Animal Classifier--Project

   Create a conda environment:
bash
Copy code
# Create the conda environment
conda create -n cnncls python=3.8 -y

# Activate the conda environment
conda activate cnncls
Save to grepper
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Save to grepper
Run the app:
bash
Copy code
python app.py
Save to grepper
Open your web browser and navigate to your local host and port to access the app.
DVC Commands
For managing the data and model pipeline, you can use the following DVC commands:

Initialize DVC:
bash
Copy code
dvc init
Save to grepper
Reproduce the pipeline:
bash
Copy code
dvc repro
Save to grepper
Generate the pipeline DAG (Directed Acyclic Graph):
bash
Copy code
dvc dag
Save to grepper
AWS Deployment with GitHub Actions
To deploy the Animal Classifier App on AWS using GitHub Actions, follow these steps:

Login to the AWS console.

Create an IAM user for deployment with specific access permissions, including EC2 and ECR.

Create an ECR repository to store the Docker image.

Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken
Create an EC2 machine (Ubuntu).

Install Docker on the EC2 machine.

Configure the EC2 machine as a self-hosted runner.

Set up the following GitHub secrets in your repository:

AWS_ACCESS_KEY_ID: Your AWS access key ID.
AWS_SECRET_ACCESS_KEY: Your AWS secret access key.
AWS_REGION: The AWS region (e.g., us-east-1).
AWS_ECR_LOGIN_URI: The login URI for the ECR (e.g., 566373416292.dkr.ecr.ap-south-1.amazonaws.com).
ECR_REPOSITORY_NAME: The name of the ECR repository (e.g., simple-app).
Azure Deployment with GitHub Actions
To deploy the Animal Classifier App on Azure using GitHub Actions, follow these steps:

Save the following passphrase for later use:

css
Copy code
s3cEZKH5yytiVnJ3h+eI3qhhzf9q1vNwEi6+q+WGdd+ACRCZ7JD6
Set up the necessary configurations and secrets in your GitHub Actions workflow for Azure deployment.

These deployment options enable you to deploy the Animal Classifier App to your preferred cloud environment seamlessly.

Conclusion
The Animal Classifier App is a remarkable tool for classifying cat and dog images with exceptional accuracy. Its ease of use, along with the ability to deploy on AWS or Azure, makes it a versatile solution for various applications. Try it out today and witness the power of AI in animal classification!
