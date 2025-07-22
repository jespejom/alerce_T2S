"""
HTML Diff utility for highlighting differences between text content
"""

from difflib import SequenceMatcher
import html

def get_html_diff(a, b):
    """
    Generate a side-by-side HTML diff of two strings with character-level highlighting.
    
    Args:
        a (str): The first string to compare
        b (str): The second string to compare
        
    Returns:
        tuple: (left_html, right_html) - HTML content for the left and right sides
    """
    # Split both inputs into lines
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    
    # Create a sequence matcher for the lines
    line_matcher = SequenceMatcher(None, a_lines, b_lines)
    
    left_html = []
    right_html = []
    
    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        if tag == 'equal':
            # Lines are identical - add them without highlighting
            for line in a_lines[i1:i2]:
                left_html.append(f'<div class="diff-line">{html.escape(line)}</div>')
            for line in b_lines[j1:j2]:
                right_html.append(f'<div class="diff-line">{html.escape(line)}</div>')
                
        elif tag == 'replace':
            # Lines were replaced - use character-level diff for these lines
            for k in range(min(i2-i1, j2-j1)):
                # Compare this pair of lines with character-level matching
                char_diff_left, char_diff_right = get_character_diff(
                    a_lines[i1+k], b_lines[j1+k]
                )
                left_html.append(f'<div class="diff-line">{char_diff_left}</div>')
                right_html.append(f'<div class="diff-line">{char_diff_right}</div>')
            
            # Handle any remaining lines
            for line in a_lines[i1+min(i2-i1, j2-j1):i2]:
                left_html.append(f'<div class="diff-line diff-removed">{html.escape(line)}</div>')
                right_html.append('<div class="diff-line diff-empty">&nbsp;</div>')
                
            for line in b_lines[j1+min(i2-i1, j2-j1):j2]:
                left_html.append('<div class="diff-line diff-empty">&nbsp;</div>')
                right_html.append(f'<div class="diff-line diff-added">{html.escape(line)}</div>')
                
        elif tag == 'delete':
            # Lines in a but not in b
            for line in a_lines[i1:i2]:
                left_html.append(f'<div class="diff-line diff-removed">{html.escape(line)}</div>')
                right_html.append('<div class="diff-line diff-empty">&nbsp;</div>')
                
        elif tag == 'insert':
            # Lines in b but not in a
            for line in b_lines[j1:j2]:
                left_html.append('<div class="diff-line diff-empty">&nbsp;</div>')
                right_html.append(f'<div class="diff-line diff-added">{html.escape(line)}</div>')
    
    return left_html, right_html

def get_character_diff(a, b):
    """
    Generate HTML with character-level highlighting for two lines
    
    Args:
        a (str): First line to compare
        b (str): Second line to compare
        
    Returns:
        tuple: (left_html, right_html) HTML content with character-level diffs
    """
    matcher = SequenceMatcher(None, a, b)
    
    left_fragments = []
    right_fragments = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Unchanged text
            fragment = html.escape(a[i1:i2])
            left_fragments.append(fragment)
            right_fragments.append(fragment)
            
        elif tag == 'replace':
            # Text was replaced - highlight differences
            left_fragments.append(f'<span class="char-removed">{html.escape(a[i1:i2])}</span>')
            right_fragments.append(f'<span class="char-added">{html.escape(b[j1:j2])}</span>')
            
        elif tag == 'delete':
            # Text in a but not in b
            left_fragments.append(f'<span class="char-removed">{html.escape(a[i1:i2])}</span>')
            
        elif tag == 'insert':
            # Text in b but not in a
            right_fragments.append(f'<span class="char-added">{html.escape(b[j1:j2])}</span>')
    
    return ''.join(left_fragments), ''.join(right_fragments)
