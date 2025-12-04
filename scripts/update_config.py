#!/usr/bin/env python3
import sys
import re

def update_config(path, section, key, value):
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    current_section = None
    key_regex = re.compile(r'^(\s*)' + re.escape(key) + r':\s*(.*?)$')
    section_regex = re.compile(r'^(\w+):\s*$')
    
    updated = False

    for line in lines:
        # Detect section change (top-level keys)
        m_sec = section_regex.match(line)
        if m_sec:
            current_section = m_sec.group(1)
        
        # If inside target section, look for key
        if current_section == section:
            m_key = key_regex.match(line)
            if m_key:
                indent = m_key.group(1)
                # Check if it's the correct indentation (e.g. 2 spaces for first level)
                # We accept any indentation that is strictly greater than the section's indentation (0)
                if len(indent) > 0:
                    # Preserve comments if any
                    original_val_part = m_key.group(2)
                    comment = ""
                    # Simple comment detection: starting with #
                    # This is naive but works for typical config files.
                    # If the value itself contains #, this might break, but scalars usually don't.
                    # A better way is to split by " # " or similar, but let's just replace the value part.
                    
                    # If the regex matched "value # comment", group 2 is "value # comment"
                    # We want to replace "value" with new value.
                    
                    # Split by first hash
                    parts = original_val_part.split('#', 1)
                    if len(parts) > 1:
                        comment = " #" + parts[1]
                    
                    # Construct new line
                    new_line = f"{indent}{key}: {value}{comment}\n"
                    new_lines.append(new_line)
                    updated = True
                    continue

        new_lines.append(line)

    if not updated:
        # If not found, we might want to error out or append?
        # For this script, let's print a warning but not crash
        print(f"Warning: Key '{key}' in section '{section}' not found in {path}", file=sys.stderr)
    
    with open(path, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: update_config.py <file> <section> <key> <value>", file=sys.stderr)
        sys.exit(1)
    
    path = sys.argv[1]
    section = sys.argv[2]
    key = sys.argv[3]
    value = sys.argv[4]
    
    # Special handling for updating ALL random_seeds if section is 'ALL'
    if section == 'ALL' and key == 'random_seed':
        update_config(path, 'nodes', 'random_seed', value)
        update_config(path, 'workload', 'random_seed', value)
    else:
        update_config(path, section, key, value)




