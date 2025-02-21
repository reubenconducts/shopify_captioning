import pandas as pd
import openai
import ast
from openai import OpenAI
import os
import argparse
import yaml

from log_utils import APILogger
from math import ceil
from typing import Dict, Union, Optional

def product_context(
    product_title: str = None,
    vendor: Optional[str] = None,
    category: Optional[str] = None,
    product_type: Optional[str] = None,
    store_context: Optional[str] = None,
    ) -> Union[str, None]:
    
    context = ""
    
    if product_title is not None:
        context += f'The product is {product_title}. '
    if vendor is not None:
        context += f'The vendor of the product is {vendor}. '
    if category is not None:
        context += f'The category of the product is {category}. '
    if product_type is not None:
        context += f'The type of the product is {product_type}. '
    if store_context is not None:
        context += f'Some helpful context about the store is {store_context}.'
        
    if context == "":
        print("Please ensure that at least one of product_title, vendor, category, and store_context is not None.")
        return
    
    return context
            
class CaptionGenerator:
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o-mini",
                 log_file: Optional[str] = None,
                 ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = APILogger(log_file)
        self.batch_id = 0
        
    
    def generate_text(
        self,
        model: str = "gpt-4o-mini",
        image_url: str = None, 
        context: str = None,
        ) -> Dict[str, Union[str, None]]:
        """Generate text with image assistance"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant that generates accurate alt text for product images for screen readers and helpful search engine optimization (SEO)."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Generate an SEO title and SEO description for this product. If an image is present, use the image to accurately generate alt text for the image, and use the image to assist in SEO generation. {context}. Format your response as a python dictionary with keys 'alt text', 'SEO title', and 'SEO description'. Do not bother adding the relevant python markdown, merely return the dictionary as you would write it into a python program. Please keep the alt text under 512 characters, the SEO title under 60 characters, and the SEO description under 155 characters. Use helpful buzzwords where relevant in the SEO fields. Keep the alt text very descriptive and accurate, without many buzzwords."},
                ]}
            ]
            if image_url is not None:
                messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
            )
            
            gen_text = response.choices[0].message.content
            gen_dict = ast.literal_eval(gen_text)
            alt_text = gen_dict['alt text'] if image_url is not None else None
            
            result = {
                "Image Alt Text": alt_text,
                "SEO Title": gen_dict['SEO title'],
                "SEO Description": gen_dict['SEO description']
            }
            
            log_data = {
                "image_url": image_url,
                "context": context,
                "response": result,
            }
            self.logger.log(self.batch_id, log_data)
            
            return result
            
        except Exception as e:
            print(f"Error generating text with image: {e}")
            # Fall back to text-only generation
            return {
                "Image Alt Text": "Alt text unavailable",
                "SEO Title": "SEO title unavailable",
                "SEO Description": "SEO description unavailable"
            }


    def process_csv(
        self,
        path_to_input_csv: str, 
        path_to_output_csv: str = None,
        batch_size: int = 50, 
        start_index: int = 0,
        use_vendor: bool = True,
        use_category: bool = True,
        use_type: bool = True,
        store_context: Optional[str] = None,
        ) -> int:
        """
        Process CSV in batches, returning the index of the last processed row
        """
        df = pd.read_csv(path_to_input_csv)
        print(f"Initial DataFrame shape: {df.shape}")

        # Fill in missing titles, vendors, and types using Handle mapping
        cols_to_fill = ['Title', 'Vendor', 'Product Category', 'Type']
        for col in cols_to_fill:
            handle_to_value = (df.dropna(subset=[col])
                            .drop_duplicates('Handle')
                            .set_index('Handle')[col]
                            .to_dict())
            df[col] = df[col].fillna(df['Handle'].map(handle_to_value))
        print("Filled in missing titles, vendors, categories, and types")

        # Create output path
        output_path = path_to_output_csv if path_to_output_csv is not None else "generated_" + path_to_input_csv
        
        # Load existing progress if file exists
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            df.update(existing_df)
            print(f"Loaded existing progress from {output_path}")

        df_with_images = df.dropna(subset=['Image Src'])
        print(f"Number of rows with images: {len(df_with_images)}")

        end_index = min(start_index + batch_size, len(df_with_images))
        batch_df = df_with_images.iloc[start_index:end_index].copy()
        
        # Process each row individually
        results = []
        for idx, row in batch_df.iterrows():
            print(f"Processing row {idx}")
            
            product_title=row["Title"]
            image_url = row["Image Src"]
            vendor = row["Vendor"] if use_vendor is True else None
            product_type = row["Type"] if use_type is True else None
            category = row["Product Category"] if use_category is True else None
            
            context = product_context(
                product_title=product_title,
                vendor=vendor,
                product_type=product_type,
                category=category,
                store_context=store_context,
            )
            # breakpoint()
            result = self.generate_text(
                image_url=row["Image Src"],
                context=context,
            )
                
            print(f"Result: {result}")
            results.append(result)
        
        # Convert results to DataFrame and ensure column order
        results_df = pd.DataFrame(results, index=batch_df.index)
        print(f"Results DataFrame shape: {results_df.shape}")
        
        expected_columns = ["Image Alt Text", "SEO Title", "SEO Description"]
        
        # Verify we have all expected columns
        if not all(col in results_df.columns for col in expected_columns):
            raise ValueError("Missing expected columns in results")
        
        for col in expected_columns:
            print(f"Updating column {col}")
            batch_df.loc[:, col] = results_df[col].values
            df.loc[batch_df.index, col] = batch_df[col]
            
        # Clean up duplicate handles by keeping only the first occurrence's metadata
        columns_to_clean = ['Title', 'Vendor', 'Type', 'Product Category', 'SEO Title', 'SEO Description']
        duplicate_mask = df.duplicated('Handle', keep='first')
        df.loc[duplicate_mask, columns_to_clean] = None
        
        # Save progress
        df.to_csv(output_path, index=False)
        
        print(f"Processed rows {start_index} to {end_index-1}")
        return end_index

def load_config(config_file: str) -> Dict[str, any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description='Generate alt text and SEO descriptions for product images.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    api_key: str = config.get("api_key")
    model: str = config.get("model", "gpt-4o-mini")
    log_file: Optional[str] = config.get("log_file")
    input_csv: str = config.get("input_csv")
    output_csv: Optional[str] = config.get("output_csv")
    use_vendor: bool = config.get("use_vendor", True)
    use_category: bool = config.get("use_category", True)
    use_type: bool = config.get("use_type", True)
    store_context: Optional[str] = config.get("store_context")
    batch_size: int = config.get("batch_size", 1)
    repeat: Optional[int] = config.get("repeat")
    start_index: Optional[int] = config.get("start_index")
    
    length = len(pd.read_csv(input_csv))
    repeat = repeat if repeat is not None else ceil(length / batch_size)
    start_index = start_index if start_index is not None else 0
    
    generator = CaptionGenerator(api_key=api_key, model=model, log_file=log_file)
    batch_id = generator.logger.start_new_batch()
    
    for _ in range(repeat):
        try:
            last_processed = generator.process_csv(
                input_csv,
                output_csv,
                batch_size,
                start_index,
                use_vendor,
                use_category,
                use_type,
                store_context,
            )
            print(f"Successfully processed up to row {last_processed}")
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"Last successful row: {start_index}")
        start_index += batch_size

if __name__ == "__main__":
    main()