# Shopify Product Caption Generator

This tool helps you automatically generate high-quality alt text and SEO descriptions for your Shopify products using AI. It processes your product CSV export and adds:
- Image alt text for accessibility
- SEO titles optimized for search engines
- SEO descriptions optimized for search engines



## Quick Start

1. Export your products from Shopify:
   - Go to Products in your Shopify admin
   - Click "Export" and choose "All products"
   - Save the CSV file somewhere on your computer
   - **NOTE: Make sure that nobody is making edits to the items in this CSV while you are running this script**

2. Create a config file named `config.yaml` with your settings:
   ```yaml
   api_key: "your-openai-api-key"
   input_csv: "path/to/your/shopify-export.csv"
   output_csv: "path/to/save/results.csv"
   store_context: "Brief description of what your store sells"
   batch_size: 10  # Number of products to process at once
   ```

3. Run the tool via the command line:
   ```bash
   python generate_text.py --config config.yaml
   ```
   (Change `config.yaml` to the name of your configuration file)

4. Import the results back to Shopify:
   - Go to Products in your Shopify admin
   - Click "Import"
   - Upload the generated CSV file
   - Click "overwrite products with matching handles." Unclick "publish new products to all sales channels," unless you would like to.
   - For extremely large imports, there may be a few issues to tweak manually afterwards. 

## Requirements

- Python 3.8 or newer
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

To install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration Options

In your `config.yaml` file, you can set:

- `api_key`: Your OpenAI API key (required)
- `model`: Which OpenAI model to use (default: GPT-4o mini, recommended for speed and cost)
- `input_csv`: Path to your Shopify product export CSV (required)
- `output_csv`: Where to save the results (optional)
- `store_context`: Description of your store to help the AI understand your products
- `batch_size`: Number of products to process at once (default: 50)
- `repeat`: Number of batches to iterate through (default: enough to go through the entire file)
- `use_vendor`: Include vendor information in generation (default: true)
- `use_category`: Include product category in generation (default: true)
- `use_type`: Include product type in generation (default: true)

## Example Store Context

Good store context helps the AI generate better descriptions. Examples:

```yaml
store_context: "We sell handmade wooden puzzles and brain teasers for all ages"
```
or
```yaml
store_context: "Our store specializes in eco-friendly kitchen accessories and cooking tools"
```

## Notes

- Running this program will cost approximately 0.5 cents per image

## Troubleshooting

1. **CSV Import/Export Issues**
   - Make sure you're using a fresh export from Shopify
   - Don't modify the column names in the CSV

2. **API Key Issues**
   - Verify your OpenAI API key is correct
   - Make sure you have billing set up on your OpenAI account

3. **Processing Takes Too Long**
   - Reduce the batch size in config.yaml
   - Process your catalog in smaller chunks

## Need Help?

If you run into issues:
1. Check the logs folder for detailed error messages
2. Make sure your CSV file has the required columns (Title, Image Src, etc.)
3. [Create an issue](link-to-your-repo/issues) on GitHub

## To-do

These functionalities are down the road:
- Selectively ignore certain ~~vendors~~, product categories, types, etc.
- Implement choices for other LMs (Claude, Gemini)
- Make API call more efficient (currently, some generated text is thrown out, e.g. for items with multiple images, SEO titles and descriptions are generated for every image but only saved for the first one).
- Parse product page HTML and use for more accurate SEO and captioning.
- Implement functionality to create HTML code and descriptions for product page from images and other data.
- (Distant future) integrate the Shopify API to download and upload items automatically, allowing for automated captioning at regular intervals.
- (More distant future) GUI for ease of use