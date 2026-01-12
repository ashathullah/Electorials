"""
Script to remove specific files from metadata and voters directories
"""

import os

# List of document IDs to remove
DOCUMENTS_TO_REMOVE = [
    'Tamil Nadu-(S22)_Sivaganga-(AC186)_227',
    'Tamil Nadu-(S22)_Sivaganga-(AC186)_261',
    'Tamil Nadu-(S22)_Sivaganga-(AC186)_75',
    'Tamil Nadu-(S22)_Sriperumbudur-(AC29)_336',
    'Tamil Nadu-(S22)_Sriperumbudur-(AC29)_337',
    'Tamil Nadu-(S22)_Tiruppur (South)-(AC114)_66'
]

# Directory paths
METADATA_DIR = os.path.join(os.path.dirname(__file__), 'metadata')
VOTERS_DIR = os.path.join(os.path.dirname(__file__), 'voters')


def remove_files():
    """Remove metadata and voter files for specified document IDs"""
    
    print(f"\n{'='*80}")
    print(f"🗑️  FILE REMOVAL SCRIPT")
    print(f"{'='*80}")
    print(f"📋 Files to remove: {len(DOCUMENTS_TO_REMOVE)}")
    print(f"{'='*80}\n")
    
    metadata_removed = 0
    metadata_not_found = 0
    voters_removed = 0
    voters_not_found = 0
    
    for doc_id in DOCUMENTS_TO_REMOVE:
        print(f"\n{'─'*80}")
        print(f"📄 Processing: {doc_id}")
        print(f"{'─'*80}")
        
        # Remove metadata file
        metadata_file = f"{doc_id}_metadata.json"
        metadata_path = os.path.join(METADATA_DIR, metadata_file)
        
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
                print(f"✅ Deleted metadata: {metadata_file}")
                metadata_removed += 1
            except Exception as e:
                print(f"❌ Error deleting metadata: {e}")
        else:
            print(f"⚠️  Metadata file not found: {metadata_file}")
            metadata_not_found += 1
        
        # Remove voters file
        voters_file = f"{doc_id}_voters.json"
        voters_path = os.path.join(VOTERS_DIR, voters_file)
        
        if os.path.exists(voters_path):
            try:
                os.remove(voters_path)
                print(f"✅ Deleted voters: {voters_file}")
                voters_removed += 1
            except Exception as e:
                print(f"❌ Error deleting voters: {e}")
        else:
            print(f"⚠️  Voters file not found: {voters_file}")
            voters_not_found += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"🏁 REMOVAL COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Metadata files:")
    print(f"   ✅ Deleted: {metadata_removed}")
    print(f"   ⚠️  Not found: {metadata_not_found}")
    print(f"\n📊 Voters files:")
    print(f"   ✅ Deleted: {voters_removed}")
    print(f"   ⚠️  Not found: {voters_not_found}")
    print(f"\n📊 Total files deleted: {metadata_removed + voters_removed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    remove_files()
