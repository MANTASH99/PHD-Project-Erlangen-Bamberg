import json
import base64
import os

def embed_images_in_json(input_json_path, project_root, output_json_path):
    """
    Embed images as base64 in JSON for Label Studio import
    
    Args:
        input_json_path: Path to your original JSON file
        project_root: Root folder of your project (where shap_results folder is)
        output_json_path: Where to save the Label Studio JSON
    """
    
    print("="*60)
    print("EMBEDDING IMAGES INTO JSON FOR LABEL STUDIO")
    print("="*60)
    
    # Load your original JSON
    print(f"\n📂 Loading JSON from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Found {len(data)} items")
    
    # Debug: Show first item structure
    if len(data) > 0:
        print(f"\n🔍 First item keys: {list(data[0].keys())}")
        print(f"🔍 First item image field: '{data[0].get('image', 'MISSING')}'")
        if data[0].get('image'):
            test_path = os.path.join(project_root, data[0]['image'])
            print(f"🔍 Testing first image path: {test_path}")
            print(f"🔍 Does it exist? {os.path.exists(test_path)}")
    print()
    
    # Process each item
    label_studio_data = []
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(data, 1):
        image_rel_path = item.get('image', '')
        print(f"Processing {i}/{len(data)}: {image_rel_path}...", end='\r')
        
        if not image_rel_path:
            print(f"\n⚠️  Item {i} has no image, skipping")
            fail_count += 1
            continue
        
        # The image path in JSON is relative to project root
        # e.g., "shap_results/bert-base-uncased-emotions/png/sample_43757_1_bar.png"
        image_path = os.path.join(project_root, image_rel_path)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"\n❌ Image not found: {image_path}")
            print(f"   Tried relative path: {image_rel_path}")
            fail_count += 1
            continue
        
        # Read and encode image to base64
        try:
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
                base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create data URL (this embeds the image)
            image_data_url = f"data:image/png;base64,{base64_string}"
            
            # Create Label Studio task format
            task = {
                "data": {
                    "image": image_data_url,  # Embedded image!
                    "text": item['text'],
                    "true_label": item['true_label'],
                    "pred_label": item['pred_label'],
                    "confidence": round(item['confidence'], 4),
                    "global_id": item['global_id'],
                    "model": item.get('model', 'unknown'),
                    # Include tokens and SHAP values for reference
                    "tokens": item.get('tokens', []),
                    "shap_values": item.get('shap_values', [])
                }
            }
            
            label_studio_data.append(task)
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {image_rel_path}: {e}")
            fail_count += 1
            continue
    
    print(f"\n\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successfully embedded: {success_count} images")
    print(f"✗ Failed: {fail_count} images")
    print(f"📋 Total tasks created: {len(label_studio_data)}")
    
    if len(label_studio_data) == 0:
        print("\n❌ ERROR: No tasks were created!")
        print("   Check the error messages above to see why images failed to load.")
        print("   Common issues:")
        print("   - Image paths in JSON don't match actual file locations")
        print("   - PROJECT_ROOT is set incorrectly")
        print("   - Images don't exist at the specified paths")
        return None
    
    # Save output
    print(f"\n💾 Saving to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(label_studio_data, f, indent=2, ensure_ascii=False)
    
    # Show file size
    file_size_bytes = os.path.getsize(output_json_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"✓ Saved successfully!")
    print(f"📊 File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    if file_size_mb > 50:
        print("\n⚠️  Large file warning:")
        print("   This file is quite large. It will work but may be slow.")
        print("   Consider processing fewer samples or splitting into batches.")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Start Label Studio:")
    print("   $ label-studio start")
    print("")
    print("2. Open http://localhost:8080 in your browser")
    print("")
    print("3. Create a new project")
    print("")
    print(f"4. Import the file: {output_json_path}")
    print("")
    print("5. Set up your labeling interface")
    print(f"{'='*60}\n")
    
    return label_studio_data


if __name__ == "__main__":
    # ============================================
    # CONFIGURE THESE PATHS FOR YOUR SETUP
    # ============================================
    
    # Path to your annotations JSON
    INPUT_JSON = "../shap_results/bert-base-uncased-emotions/annotations.json"
    
    # Path to your PROJECT ROOT (where shap_results folder is located)
    # The JSON contains relative paths like "shap_results/model/png/image.png"
    # so we need the root folder to resolve them
    PROJECT_ROOT = ".."  # If running from scripts folder
    
    # Or use absolute path:
    # PROJECT_ROOT = "/home/iv93baik/PHD/PHD-Project-Erlangen-Bamberg/emotion_project"
    
    OUTPUT_JSON = "label_studio_data_bert.json"  # Output file for Label Studio
    
    # ============================================
    
    # Debug: Show current working directory
    print(f"🔍 Current directory: {os.getcwd()}")
    print(f"🔍 Project root: {os.path.abspath(PROJECT_ROOT)}")
    print(f"🔍 Looking for JSON: {INPUT_JSON}")
    print(f"🔍 JSON absolute path: {os.path.abspath(INPUT_JSON)}")
    print()
    
    # Check if input files exist
    if not os.path.exists(INPUT_JSON):
        print(f"❌ ERROR: Cannot find input file: {INPUT_JSON}")
        print(f"   Absolute path checked: {os.path.abspath(INPUT_JSON)}")
        print("\n👉 Please update INPUT_JSON in the script to point to your JSON file")
        print("\nTip: Try using absolute path starting with /home/...")
        exit(1)
    
    if not os.path.exists(PROJECT_ROOT):
        print(f"❌ ERROR: Cannot find project root: {PROJECT_ROOT}")
        print(f"   Absolute path checked: {os.path.abspath(PROJECT_ROOT)}")
        print("\n👉 Please update PROJECT_ROOT in the script")
        exit(1)
    
    # Run the conversion
    embed_images_in_json(INPUT_JSON, PROJECT_ROOT, OUTPUT_JSON)