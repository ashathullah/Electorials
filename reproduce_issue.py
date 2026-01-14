
import re

class MockOCRProcessor:
    def _fix_ocr_digits(self, text: str) -> str:
        """
        Fix common OCR letter/digit confusions in text.
        """
        if not text:
            return text
        
        result = []
        chars = list(text)
        
        for i, c in enumerate(chars):
            prev_is_digit = (i > 0 and chars[i-1].isdigit())
            next_is_digit = (i < len(chars) - 1 and chars[i+1].isdigit())
            
            # Only convert O->0 or I->1 if surrounded by or adjacent to digits
            if c == 'O' and (prev_is_digit or next_is_digit):
                result.append('0')
            elif c == 'I' and (prev_is_digit or next_is_digit):
                result.append('1')
            else:
                result.append(c)
        
        return "".join(result)

    def _clean_house_number(self, raw_text: str) -> str:
        """
        Clean and correct OCR'd house number text. (COPIED FROM UPDATED CODE)
        """
        if not raw_text:
            return ""
        
        tokens = raw_text.split()
        
        for t in tokens:
            t = t.strip(".,()[]")
            
            # Apply digit fixes first (O->0, I->1) - critical for O35-1 pattern
            t = self._fix_ocr_digits(t)
            
            if not re.search(r"\d", t):
                continue
            
            # First, handle the noise prefix pattern like "6L283", "6GL324"
            noise_prefix_match = re.match(r"^(\d{1,2})([A-Z]{1,2})(\d+.*)$", t)
            if noise_prefix_match:
                real_part = noise_prefix_match.group(3)
                if len(real_part) <= 20:
                    return real_part
            
            # Handle the leading-0-is-actually-a-letter pattern
            # e.g., "0035-1" -> "D35-1", "035-1" -> "D35-1"
            leading_zero_match = re.match(r"^(0{1,2})(\d{1,})([-/]\d+.*)?$", t)
            if leading_zero_match:
                leading_zeros = leading_zero_match.group(1)
                remaining_digits = leading_zero_match.group(2)
                suffix = leading_zero_match.group(3) or ""
                
                # Convert leading zero(s) to 'D' if followed by digits
                # This fixes '035-1' -> 'D35-1'
                if len(remaining_digits) >= 1:
                     return f"D{remaining_digits}{suffix}"
            
            # Check for valid alphanumeric patterns
            # Relaxed regex to capture "11-A", "12/4-B", etc.
            # Must start with alphanum, can have internal - or /
            if re.match(r"^[A-Z0-9]+(?:[-/][A-Z0-9]+)*$", t):
                # Reject PIN codes (6 digits starting with 6, no separators)
                if len(t) == 6 and t.isdigit() and t.startswith("6"):
                    continue
                    
                if len(t) <= 20:
                    return t
            
            # Fallback: Extract longest digit sequence
            digit_sequences = re.findall(r"\d+", t)
            if digit_sequences:
                cleaned = max(digit_sequences, key=len)
                if len(cleaned) == 6 and cleaned.startswith("6"):
                    continue
                if len(cleaned) <= 4:
                    return cleaned
        
        return ""

processor = MockOCRProcessor()

test_cases = [
    ("035-1", "D35-1"),        # User Case 1
    ("O35-1", "D35-1"),        # Common OCR error
    ("0035-1", "D35-1"),       # Original design
    ("11-", "11"),             # User Case 2 (clean trailing dash)
    ("11-A", "11-A"),          # Complex number support
    ("12/4-B", "12/4-B"),      # Complex number support
    ("Flat 12A", "12A"),       # Token selection
    ("No: 15", "15"),          # Token selection
    ("6L283", "283"),          # Noise removal
    ("D35-1", "D35-1"),        # Already correct
    ("123", "123")             # Simple
]

print("Running Tests...")
failed = False
for input_val, expected in test_cases:
    result = processor._clean_house_number(input_val)
    if result != expected:
        print(f"FAIL: Input '{input_val}' => Got '{result}', Expected '{expected}'")
        failed = True
    else:
        print(f"PASS: '{input_val}' => '{result}'")

if not failed:
    print("\nAll tests passed!")
