import os
import sys
import argparse
import subprocess

MODELS = [
    'A_BiLSTM_Attn',      
    'B_TextCNN',          
    'C_BiGRU',            
]

def train_all_models(data_path, splits_dir, artifact_dir, seed):
    print("="*70)
    print("TRAINING ALL 3 MODELS")
    print("="*70)
    print(f"\nData: {data_path}")
    print(f"Splits: {splits_dir}")
    print(f"Output: {artifact_dir}")
    print(f"Seed: {seed}")
    print(f"\nModels to train: {len(MODELS)}")
    for i, model_key in enumerate(MODELS, 1):
        student = model_key[0]
        print(f"  {i}. {model_key} (Student {student})")
    print()

    results = []

    for i, model_key in enumerate(MODELS, 1):
        print(f"\n{'='*70}")
        print(f"Model {i}/{len(MODELS)}: {model_key}")
        print(f"{'='*70}\n")

        cmd = [
            'python', 'Training/train_model.py',
            '--model_key', model_key,
            '--data', data_path,
            '--splits', splits_dir,
            '--out', artifact_dir,
            '--seed', str(seed)
        ]

        try:
            result = subprocess.run(cmd, check=True)
            results.append((model_key, True))
            print(f"\n✓ {model_key} completed successfully")
        except subprocess.CalledProcessError as e:
            results.append((model_key, False))
            print(f"\n✗ {model_key} failed: {e}")
        except KeyboardInterrupt:
            print(f"\n⚠ Training interrupted at {model_key}")
            break

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_key, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {model_key}")
    print("="*70)

    successful = sum(1 for _, success in results if success)
    print(f"\n{successful}/{len(MODELS)} models trained successfully")
    print(f"\nAll artifacts saved to: {artifact_dir}")
    print("Each model has its own directory with all required artifacts.")

def main():
    parser = argparse.ArgumentParser(description="Train all 3 models")
    parser.add_argument('--data', type=str, default='./data/train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--splits', type=str, default='./artifacts/splits',
                       help='Directory containing split indices')
    parser.add_argument('--out', type=str, default='./artifacts',
                       help='Output directory for artifacts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.splits, 'train_idx.npy')):
        print(f"Error: Split indices not found in {args.splits}")
        print("Please run: python Training/split_data.py first")
        return

    train_all_models(args.data, args.splits, args.out, args.seed)

if __name__ == "__main__":
    main()
