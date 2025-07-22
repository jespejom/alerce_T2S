"""
Helper functions for SQL Line Comparison Widget to handle SQL parsing tasks.
These functions help parse and extract information from SQL queries, specifically focusing on JOIN and WHERE clauses.
"""

import re

def handle_complex_join_conditions(join_condition):
    """
    Split complex JOIN conditions into individual conditions based on AND/OR operators.
    
    Args:
        join_condition (str): The join condition part of a SQL query (e.g., "table1.id = table2.id AND table1.name = table2.name")
    
    Returns:
        list: A list of individual conditions
    """
    # Initialize the result list and variables for parsing
    conditions = []
    current_condition = ""
    paren_count = 0
    in_quotes = False
    quote_char = None
    
    # Add a space at the end to handle the last condition
    join_condition = join_condition.strip() + " "
    
    i = 0
    while i < len(join_condition):
        char = join_condition[i]
        
        # Handle quotes (string literals)
        if char in ["'", '"'] and (i == 0 or join_condition[i-1] != "\\"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        
        # Only process operators when not in quotes
        if not in_quotes:
            # Handle parentheses for nested conditions
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            # Check for AND/OR operators but only at the top level (paren_count == 0)
            if paren_count == 0 and i + 4 < len(join_condition):
                # Check for AND operator
                if join_condition[i:i+5].upper() == " AND " or join_condition[i:i+5].upper() == " OR  ":
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 4  # Skip past the operator
                    continue
                # Check for OR operator
                elif join_condition[i:i+4].upper() == " OR " and i + 3 < len(join_condition):
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 3  # Skip past the operator
                    continue
        
        current_condition += char
        i += 1
    
    # Add the last condition if it exists
    if current_condition.strip():
        conditions.append(current_condition.strip())
    
    return conditions

def split_join_clause(join_clause):
    """
    Split a JOIN clause into table part and ON condition part.
    
    Args:
        join_clause (str): The JOIN clause to split (e.g., "JOIN table2 ON table1.id = table2.id")
    
    Returns:
        tuple: (table_part, on_part) where table_part is the table expression and on_part is the ON condition
    """
    # Look for the ON keyword, but handle potential subqueries or table expressions
    on_pattern = re.compile(r'\bON\b', re.IGNORECASE)
    matches = list(on_pattern.finditer(join_clause))
    
    if not matches:
        # No ON clause found - may be a simple join or using clause
        using_pattern = re.compile(r'\bUSING\b', re.IGNORECASE)
        using_matches = list(using_pattern.finditer(join_clause))
        
        if using_matches:
            # Handle USING clause
            pos = using_matches[0].start()
            return join_clause[:pos].strip(), join_clause[pos:].strip()
        else:
            # No ON or USING found - treat as simple join
            return join_clause.strip(), ""
    
    # For ON clauses, find the correct match (not in a subquery or nested expression)
    for match in matches:
        pos = match.start()
        # Check if this ON is at the right level (not inside a subquery)
        text_before = join_clause[:pos]
        paren_count = text_before.count('(') - text_before.count(')')
        
        # If parentheses are balanced, this is the top-level ON
        if paren_count == 0:
            return join_clause[:pos].strip(), join_clause[pos:].strip()
    
    # If we couldn't find a proper ON clause, return the whole thing as table part
    return join_clause.strip(), ""

def extract_on_clauses_from_joins(sql_query):
    """
    Extract all ON clauses from JOIN statements in a SQL query.
    
    Args:
        sql_query (str): The SQL query to analyze
    
    Returns:
        list: A list of extracted ON clauses
    """
    # First, find all JOIN statements in the query
    join_pattern = re.compile(r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\b\s+(.*?)(?=\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET|$))', re.IGNORECASE | re.DOTALL)
    
    on_clauses = []
    
    # Preprocess to handle string literals and comments
    in_single_quote = False
    in_double_quote = False
    in_comment = False
    processed_sql = ""
    
    for i, char in enumerate(sql_query):
        if char == "'" and (i == 0 or sql_query[i-1] != '\\') and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and (i == 0 or sql_query[i-1] != '\\') and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == '-' and i + 1 < len(sql_query) and sql_query[i+1] == '-' and not in_single_quote and not in_double_quote:
            in_comment = True
        elif char == '\n' and in_comment:
            in_comment = False
        
        if not in_comment:
            processed_sql += char
    
    # Find all JOIN statements
    for match in join_pattern.finditer(processed_sql):
        join_clause = match.group(2).strip()
        
        # Extract the ON part
        _, on_part = split_join_clause(join_clause)
        
        # If we found an ON clause, process it
        if on_part and on_part.upper().startswith('ON '):
            # Remove the leading ON keyword
            on_condition = on_part[2:].strip()
            on_clauses.append(on_condition)
            
            # Also extract complex conditions (connected by AND/OR)
            individual_conditions = handle_complex_join_conditions(on_condition)
            on_clauses.extend(individual_conditions)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_on_clauses = []
    for clause in on_clauses:
        if clause not in seen:
            seen.add(clause)
            unique_on_clauses.append(clause)
    
    return unique_on_clauses
