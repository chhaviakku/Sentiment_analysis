import speech_recognition as sr
import pandas as pd
from transformers import pipeline

# Initialize the recognizer
recognizer = sr.Recognizer()

while True:
    try:
        # Open the microphone and capture audio input
        with sr.Microphone() as source:
            print("Speak something...")
            audio = recognizer.listen(source)

        # Recognize the audio and convert it to text using Google Speech Recognition
        text = recognizer.recognize_google(audio)
        print(f"Transcription: {text}")

        # Read the existing CSV file or create a new one if it doesn't exist
        try:
            df = pd.read_csv("input.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Statements"])

        # Append the new transcription to the DataFrame
        df = df._append({"Statements": text}, ignore_index=True)

        # Save the updated DataFrame to the CSV file in append mode
        df.to_csv("input.csv", mode='a', header=False, index=False)
        print("Transcription appended to 'input.csv'")

        # Now, use the machine learning model to classify the text from the CSV file
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)

        # Process the text from the CSV file
        model_outputs = classifier(df["Statements"].tolist())

        # Create a new DataFrame to store the scores
        scores_df = pd.DataFrame(model_outputs)

        # Concatenate the original DataFrame with the scores DataFrame
        result_df = pd.concat([df, scores_df], axis=1)

        # Save the result to a new CSV file in append mode
        result_df.to_csv('output.csv', mode='a', header=False, index=False)
        print("Classification results appended to 'output.csv'")

        # Print the classification results
        for i, row in result_df.iterrows():
            statement = row['Statements']
            print(f"\nStatement {i + 1}: {statement}")

            print("Emotions:")
            for col_idx in range(5):
                emotion_info = row[col_idx]
                label = emotion_info['label']
                score = emotion_info['score']
                print(f"{label}: {score:.4f}")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    # Add an option to exit the loop
    exit_command = input("Write 'exit' to stop or press Enter to continue: ")
    if exit_command.lower() == 'exit':
        break  # Exit the loop when 'exit' is written or Enter is pressed

print("Program terminated.")

