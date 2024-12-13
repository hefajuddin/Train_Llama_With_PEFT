from transformers import pipeline

def main():
    while True:
        data=input("Input to generate from Rapunzel story: ")
        if data !="":
            break

    model_name = "hefajuddin/Rapunzel_Story_Gen"
    generator = pipeline("text-generation", model=model_name)
    text = data
    output = generator(text, max_length=100, num_return_sequences=1)

    print("\033[92m" + output[0]["generated_text"]+ "\033[0m")

if __name__ == "__main__":
    main()