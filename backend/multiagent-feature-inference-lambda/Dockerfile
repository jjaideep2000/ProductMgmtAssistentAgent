FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

# Copy requirements and install
COPY app/requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the function code
COPY app/lambda_function.py .

# Set the handler
CMD ["lambda_function.lambda_handler"]
