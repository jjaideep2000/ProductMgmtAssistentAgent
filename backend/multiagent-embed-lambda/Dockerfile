# Use official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

# Copy dependency file first (to benefit from Docker caching)
COPY app/requirements.txt .

# Upgrade pip and install dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the actual Lambda code
COPY app .

# Command to run the Lambda function
CMD ["lambda_function.lambda_handler"]