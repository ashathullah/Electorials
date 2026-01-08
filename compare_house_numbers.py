import json
from typing import Dict, List, Tuple

def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_house_numbers(page_wise_file: str, ai_debug_file: str, page_name: str) -> List[dict]:
    """Compare house numbers between page_wise and AI debug data."""
    
    # Load data
    page_wise = load_json(page_wise_file)
    ai_debug = load_json(ai_debug_file)
    
    # Create lookup dictionary from AI data
    ai_lookup = {item['serial_no']: item['house_no'] for item in ai_debug}
    
    # Compare and collect results
    comparisons = []
    
    for record in page_wise['records']:
        serial_no = record['serial_no']
        page_wise_house = record['house_no']
        ai_house = ai_lookup.get(serial_no, "NOT_FOUND")
        
        match = page_wise_house == ai_house
        
        comparisons.append({
            'serial_no': serial_no,
            'page_wise_house_no': page_wise_house,
            'ai_debug_house_no': ai_house,
            'match': match,
            'epic_no': record['epic_no'],
            'name': record['name']
        })
    
    return comparisons

def print_comparison_report(comparisons: List[dict], page_name: str):
    """Print formatted comparison report."""
    
    print(f"\n{'='*100}")
    print(f"HOUSE NUMBER COMPARISON REPORT - {page_name}")
    print(f"{'='*100}\n")
    
    # Count matches and mismatches
    matches = sum(1 for c in comparisons if c['match'])
    mismatches = sum(1 for c in comparisons if not c['match'])
    
    print(f"Total Records: {len(comparisons)}")
    print(f"Matches: {matches} ({matches/len(comparisons)*100:.1f}%)")
    print(f"Mismatches: {mismatches} ({mismatches/len(comparisons)*100:.1f}%)")
    print(f"\n{'='*100}\n")
    
    # Print all comparisons
    print(f"{'Serial':<8} {'EPIC No':<15} {'Name':<25} {'Page-Wise':<15} {'AI Debug':<15} {'Match':<8}")
    print(f"{'-'*100}")
    
    for comp in comparisons:
        match_symbol = "YES" if comp['match'] else "NO"
        print(f"{comp['serial_no']:<8} {comp['epic_no']:<15} {comp['name']:<25} {comp['page_wise_house_no']:<15} {comp['ai_debug_house_no']:<15} {match_symbol:<8}")
    
    # Print mismatches separately if any
    if mismatches > 0:
        print(f"\n{'='*100}")
        print(f"MISMATCHES ONLY ({mismatches} records)")
        print(f"{'='*100}\n")
        print(f"{'Serial':<8} {'EPIC No':<15} {'Name':<25} {'Page-Wise':<15} {'AI Debug':<15}")
        print(f"{'-'*100}")
        
        for comp in comparisons:
            if not comp['match']:
                print(f"{comp['serial_no']:<8} {comp['epic_no']:<15} {comp['name']:<25} {comp['page_wise_house_no']:<15} {comp['ai_debug_house_no']:<15}")

def main():
    """Main comparison function."""
    
    base_path = r"e:\Raja_mohaemd\projects\Electorials\extracted\2025-EROLLGEN-S22-114-FinalRoll-Revision1-TAM-1-WI\output"
    
    # Page 042
    print("\n" + "="*100)
    print("COMPARING PAGE 042")
    print("="*100)
    
    page_042_comparisons = compare_house_numbers(
        f"{base_path}\\page_wise\\page-042.json",
        f"{base_path}\\debug\\id_extract_page-042.json",
        "Page 042"
    )
    print_comparison_report(page_042_comparisons, "PAGE 042")
    
    # Page 043
    print("\n\n" + "="*100)
    print("COMPARING PAGE 043")
    print("="*100)
    
    page_043_comparisons = compare_house_numbers(
        f"{base_path}\\page_wise\\page-043.json",
        f"{base_path}\\debug\\id_extract_page-043.json",
        "Page 043"
    )
    print_comparison_report(page_043_comparisons, "PAGE 043")
    
    # Overall summary
    all_comparisons = page_042_comparisons + page_043_comparisons
    total_matches = sum(1 for c in all_comparisons if c['match'])
    total_mismatches = sum(1 for c in all_comparisons if not c['match'])
    
    print(f"\n\n{'='*100}")
    print(f"OVERALL SUMMARY (Both Pages)")
    print(f"{'='*100}")
    print(f"Total Records: {len(all_comparisons)}")
    print(f"Total Matches: {total_matches} ({total_matches/len(all_comparisons)*100:.1f}%)")
    print(f"Total Mismatches: {total_mismatches} ({total_mismatches/len(all_comparisons)*100:.1f}%)")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
