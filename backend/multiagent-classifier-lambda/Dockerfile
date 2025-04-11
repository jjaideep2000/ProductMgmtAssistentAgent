FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

# Copy and install dependencies
COPY app/requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy your Lambda function code
COPY app/lambda_function.py .

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
