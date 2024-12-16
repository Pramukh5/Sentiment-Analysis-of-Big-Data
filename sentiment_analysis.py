from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, regexp_replace, trim, when, lit, rand
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import nltk
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .master("local[*]") \
    .getOrCreate()

def load_data(file_path):
    """Load the dataset with specific schema"""
    # Define the schema to match your CSV format
    schema = StructType([
        StructField("ItemID", StringType(), True),
        StructField("Sentiment", IntegerType(), True),
        StructField("SentimentText", StringType(), True)
    ])
    
    # Read CSV with schema
    df = spark.read.csv(file_path, header=True, schema=schema)
    
    # Clean the text data
    df = df.withColumn("SentimentText", 
                      trim(regexp_replace(col("SentimentText"), r'[^\w\s]', ' ')))
    
    # Prepare data for pipeline (rename columns to match pipeline)
    return df.select(
        col("SentimentText").alias("text"),
        col("Sentiment").alias("label")
    )

def create_pipeline():
    """Create the ML pipeline"""
    # Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Convert words to vectors
    countVectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", minDF=2.0)
    
    # TF-IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Logistic Regression Classifier
    lr = LogisticRegression(maxIter=20, regParam=0.01)
    
    # Create the pipeline
    return Pipeline(stages=[tokenizer, remover, countVectorizer, idf, lr])

def display_results(predictions_df, num_results, display_option='random'):
    """Display results with emojis and sentiment analysis"""
    print("\nPrediction Results:")
    print(f"Display Option: {display_option}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    
    if display_option == 'random':
        # Get random samples
        pd_df = predictions_df.orderBy(rand()).limit(num_results).toPandas()
    elif display_option == 'most_confident':
        # Get most confident predictions
        pd_df = predictions_df.orderBy(abs(col("prediction") - 0.5).desc()).limit(num_results).toPandas()
    elif display_option == 'most_positive':
        # Get most positive predictions
        pd_df = predictions_df.orderBy(col("prediction").desc()).limit(num_results).toPandas()
    elif display_option == 'most_negative':
        # Get most negative predictions
        pd_df = predictions_df.orderBy(col("prediction")).limit(num_results).toPandas()
    else:
        pd_df = predictions_df.limit(num_results).toPandas()
    
    for idx, row in pd_df.iterrows():
        sentiment_emoji = "✅" if row['prediction'] == 1.0 else "❌"
        sentiment_text = "positive" if row['prediction'] == 1.0 else "negative"
        actual_sentiment = "positive" if row['label'] == 1 else "negative"
        confidence = abs(row['prediction'] - 0.5) * 2
        
        print(f"Sample #{idx + 1}")
        print(f"Text: {row['text'][:100]}...")
        print(f"Prediction: {sentiment_emoji} {sentiment_text} (Confidence: {confidence:.2%})")
        print(f"Actual: {actual_sentiment}")
        print("-" * 100)

def plot_sentiment_distribution(predictions_df):
    """Plot the distribution of sentiments"""
    plt.figure(figsize=(10, 6))
    
    # Convert to pandas and count sentiments
    sentiment_counts = predictions_df.groupBy('prediction').count().toPandas()
    
    # Create bar plot
    plt.bar(['Negative', 'Positive'], sentiment_counts['count'])
    plt.title('Distribution of Sentiments')
    plt.ylabel('Count')
    plt.show()

def plot_confidence_histogram(predictions_df):
    """Plot histogram of prediction confidences"""
    plt.figure(figsize=(10, 6))
    
    # Calculate confidence scores using PySpark SQL functions
    from pyspark.sql.functions import abs as spark_abs
    
    confidences = predictions_df.select(
        (spark_abs(col('prediction') - 0.5) * 2).alias('confidence')
    ).toPandas()
    
    # Create histogram
    plt.hist(confidences['confidence'], bins=20)
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.show()

def plot_confusion_matrix(predictions_df):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Get predictions and actual labels
    results = predictions_df.select(['prediction', 'label']).toPandas()
    
    # Create confusion matrix
    cm = confusion_matrix(results['label'], results['prediction'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Use the actual dataset path
    file_path = r"C:\Users\pramu\Downloads\bdaproj\Sentiment Analysis Dataset.csv"
    
    # Get display preferences from user
    print("\nDisplay Options:")
    print("1. Random samples")
    print("2. Most confident predictions")
    print("3. Most positive tweets")
    print("4. Most negative tweets")
    
    while True:
        try:
            option = int(input("\nSelect display option (1-4): "))
            if 1 <= option <= 4:
                break
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")
    
    display_options = {
        1: 'random',
        2: 'most_confident',
        3: 'most_positive',
        4: 'most_negative'
    }
    
    while True:
        try:
            num_results = int(input("Enter the number of results to display (1-50): "))
            if 1 <= num_results <= 50:
                break
            else:
                print("Please enter a number between 1 and 50")
        except ValueError:
            print("Please enter a valid number")
    
    # Load the dataset
    print("\nLoading and processing data...")
    data = load_data(file_path)
    
    # Split the data
    train_data, test_data = data.randomSplit([0.8, 0.2])
    
    print(f"\nTraining Data Count: {train_data.count():,}")
    print(f"Testing Data Count: {test_data.count():,}")
    
    # Create and fit the pipeline
    print("\nTraining the model...")
    pipeline = create_pipeline()
    model = pipeline.fit(train_data)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.transform(test_data)
    
    # Calculate accuracy
    correct_predictions = predictions.filter(col("label") == col("prediction")).count()
    total_predictions = predictions.count()
    accuracy = correct_predictions / total_predictions
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Show results with emojis
    display_results(predictions, num_results, display_options[option])
    
    # After making predictions
    print("\nGenerating visualizations...")
    plot_sentiment_distribution(predictions)
    plot_confidence_histogram(predictions)
    plot_confusion_matrix(predictions)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 