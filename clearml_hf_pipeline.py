from clearml import Task, PipelineDecorator, TaskTypes
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import pandas as pd

@PipelineDecorator.component(return_values=["model_path"])
def train_model():
    task = Task.init(project_name="HuggingFace Pipeline", task_name="Train Model")
    
    # Load custom dataset
    dataset = pd.read_csv('custom_dataset.csv')
    dataset = Dataset.from_pandas(dataset)
    
    # Load pretrained model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model_path = "hf_model"
    trainer.save_model(model_path)
    
    task.close()
    return model_path

@PipelineDecorator.component()
def evaluate_model(model_path):
    task = Task.init(project_name="HuggingFace Pipeline", task_name="Evaluate Model")
    # Here you would load the model and perform evaluation
    print(f"Evaluating model from {model_path}")
    task.close()

@PipelineDecorator.pipeline(name="HuggingFace ClearML Pipeline", project="HuggingFace Pipeline Project")
def main_pipeline():
    model_path = train_model()
    evaluate_model(model_path)

if __name__ == "__main__":
    pipeline = main_pipeline()

    # Enqueue the pipeline
    task = Task.get_task(task_id=pipeline.id)
    task.set_base_task_type(TaskTypes.pipeline)
    task.execute_remotely(queue_name="services")
