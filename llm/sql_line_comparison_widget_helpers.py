"""
Helper functions for SQL Line Comparison Widget to handle SQL parsing tasks.
These functions help parse and extract information from SQL queries, specifically focusing on JOIN and WHERE clauses.
This module provides enhanced parsing capabilities for SQL queries with support for complex syntax structures.
"""

import re

def handle_complex_join_conditions(join_condition):
    """
    Split complex JOIN conditions into individual conditions based on AND/OR operators.
    Handles nested parentheses, function calls, and string literals correctly.
    
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
            if paren_count == 0:
                # Check for AND operator (with proper word boundaries)
                if (i + 4 < len(join_condition) and 
                    join_condition[i:i+5].upper() == " AND " and
                    (i == 0 or not join_condition[i-1].isalnum())):
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 5  # Skip past " AND "
                    continue
                # Check for OR operator (with proper word boundaries)
                elif (i + 3 < len(join_condition) and 
                      join_condition[i:i+4].upper() == " OR " and
                      (i == 0 or not join_condition[i-1].isalnum())):
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 4  # Skip past " OR "
                    continue
        
        current_condition += char
        i += 1
    
    # Add the last condition if it exists
    if current_condition.strip():
        conditions.append(current_condition.strip())
    
    # Post-process to handle function calls that might have been incorrectly split
    result = repair_broken_function_calls(conditions)
    
    return result

def repair_broken_function_calls(conditions):
    """
    Repair conditions that might have been incorrectly split in the middle of a function call
    
    Args:
        conditions (list): List of split conditions
        
    Returns:
        list: Repaired list with function calls properly preserved
    """
    if not conditions or len(conditions) <= 1:
        return conditions
        
    result = []
    i = 0
    
    while i < len(conditions):
        current = conditions[i]
        
        # Check if this condition has unbalanced parentheses
        open_parens = current.count('(')
        close_parens = current.count(')')
        
        if open_parens > close_parens:
            # We have unclosed parentheses - this might be a broken function call
            combined = current
            j = i + 1
            
            # Look ahead to find closing parentheses
            while j < len(conditions) and open_parens > close_parens:
                next_condition = conditions[j]
                combined += " AND " + next_condition  # Default to AND as the connector
                
                open_parens += next_condition.count('(')
                close_parens += next_condition.count(')')
                
                # If we've balanced the parentheses, we're done
                if close_parens >= open_parens:
                    result.append(combined)
                    i = j + 1
                    break
                j += 1
                
            # If we couldn't balance the parentheses, add as-is
            if open_parens > close_parens:
                result.append(current)
                i += 1
        else:
            # No imbalance, add as-is
            result.append(current)
            i += 1
    
    return result

def split_join_clause(join_clause):
    """
    Split a JOIN clause into table part and ON condition part.
    Handles subqueries, quoted strings, and nested structures correctly.
    
    Args:
        join_clause (str): The JOIN clause to split (e.g., "JOIN table2 ON table1.id = table2.id")
    
    Returns:
        tuple: (table_part, on_part) where table_part is the table expression and on_part is the ON condition
    """
    # First, scan for ON or USING keywords while respecting quotes and parentheses
    i = 0
    paren_count = 0
    in_quotes = False
    quote_char = None
    on_position = -1
    using_position = -1
    
    while i < len(join_clause):
        char = join_clause[i]
        
        # Handle quoted strings
        if char in ["'", '"'] and (i == 0 or join_clause[i-1] != "\\"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        
        # Only process keywords when not in quotes
        if not in_quotes:
            # Handle parentheses for nested expressions
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            # Check for ON keyword at the top level
            if paren_count == 0 and i + 2 < len(join_clause):
                if (join_clause[i:i+3].upper() == ' ON' and 
                    (i+3 >= len(join_clause) or not join_clause[i+3].isalnum())):
                    on_position = i
                    break
                elif (join_clause[i:i+6].upper() == ' USING' and
                      (i+6 >= len(join_clause) or not join_clause[i+6].isalnum())):
                    using_position = i
                    break
        
        i += 1
    
    # Determine how to split the clause
    if on_position >= 0:
        # Found ON clause
        return join_clause[:on_position].strip(), join_clause[on_position:].strip()
    elif using_position >= 0:
        # Found USING clause
        return join_clause[:using_position].strip(), join_clause[using_position:].strip()
    else:
        # No ON or USING found - treat as simple join
        return join_clause.strip(), ""

def extract_on_clauses_from_joins(sql_query):
    """
    Extract all ON clauses from JOIN statements in a SQL query with enhanced parsing capabilities.
    Handles complex joins, subqueries, and nested structures correctly.
    
    Args:
        sql_query (str): The SQL query to analyze
    
    Returns:
        list: A list of extracted ON clauses
    """
    # Process the SQL query to handle comments, quotes, etc.
    processed_sql = preprocess_sql(sql_query)
    
    # Extract all JOIN clauses with their contexts
    join_clauses = extract_join_clauses(processed_sql)
    
    on_clauses = []
    
    # Process each JOIN clause
    for join_clause in join_clauses:
        # Extract the ON part
        _, on_part = split_join_clause(join_clause)
        
        # If we found an ON clause, process it
        if on_part and on_part.upper().startswith('ON '):
            # Remove the leading ON keyword
            on_condition = on_part[3:].strip()
            on_clauses.append(on_condition)
            
            # Extract complex conditions (connected by AND/OR)
            individual_conditions = handle_complex_join_conditions(on_condition)
            
            # Add the individual conditions but avoid duplicating the original condition
            for condition in individual_conditions:
                if condition != on_condition:
                    on_clauses.append(condition)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_on_clauses = []
    for clause in on_clauses:
        if clause and clause not in seen:
            seen.add(clause)
            unique_on_clauses.append(clause)
    
    return unique_on_clauses

def preprocess_sql(sql_query):
    """
    Preprocess SQL query to handle comments, quotes, and other special syntax.
    
    Args:
        sql_query (str): The raw SQL query to preprocess
        
    Returns:
        str: Preprocessed SQL query with comments removed
    """
    in_single_quote = False
    in_double_quote = False
    in_comment = False
    in_block_comment = False
    processed_sql = ""
    
    i = 0
    while i < len(sql_query):
        char = sql_query[i]
        
        # Handle string literals
        if not in_comment and not in_block_comment:
            if char == "'" and (i == 0 or sql_query[i-1] != '\\') and not in_double_quote:
                in_single_quote = not in_single_quote
                processed_sql += char
            elif char == '"' and (i == 0 or sql_query[i-1] != '\\') and not in_single_quote:
                in_double_quote = not in_double_quote
                processed_sql += char
            # Handle comments
            elif not in_single_quote and not in_double_quote:
                # Handle line comments
                if char == '-' and i + 1 < len(sql_query) and sql_query[i+1] == '-':
                    in_comment = True
                    i += 1  # Skip the next dash
                # Handle block comments
                elif char == '/' and i + 1 < len(sql_query) and sql_query[i+1] == '*':
                    in_block_comment = True
                    i += 1  # Skip the next asterisk
                else:
                    processed_sql += char
            else:
                processed_sql += char
        # Handle end of comments
        elif in_comment and char == '\n':
            in_comment = False
            processed_sql += char
        elif in_block_comment and char == '*' and i + 1 < len(sql_query) and sql_query[i+1] == '/':
            in_block_comment = False
            i += 1  # Skip the next slash
        elif not in_comment and not in_block_comment:
            processed_sql += char
        
        i += 1
    
    return processed_sql

def extract_join_clauses(sql_query):
    """
    Extract all JOIN clauses from a SQL query, handling complex cases like subqueries.
    
    Args:
        sql_query (str): The SQL query to analyze
        
    Returns:
        list: A list of extracted JOIN clauses with their context
    """
    join_clauses = []
    
    # Tokenize the query to find JOIN keywords
    tokens = tokenize_sql(sql_query)
    
    # Scan for JOIN keywords and extract their clauses
    for i, token in enumerate(tokens):
        if token.upper() == 'JOIN' or token.upper().endswith(' JOIN'):
            # Found a JOIN token, extract the full clause
            join_start = i
            
            # Look for the next major clause keyword or the end
            next_join_idx = find_next_clause(tokens, i+1)
            
            if next_join_idx == -1:
                # No more clauses, extract to the end
                join_clause = ' '.join(tokens[join_start:])
            else:
                # Extract up to the next clause
                join_clause = ' '.join(tokens[join_start:next_join_idx])
            
            join_clauses.append(join_clause)
    
    return join_clauses

def tokenize_sql(sql_query):
    """
    Tokenize a SQL query into its component parts, handling quotes and parentheses.
    
    Args:
        sql_query (str): The SQL query to tokenize
        
    Returns:
        list: A list of SQL tokens
    """
    # Simple tokenization by whitespace, but keep quotes and parenthesized expressions intact
    tokens = []
    i = 0
    current_token = ""
    in_quotes = False
    quote_char = None
    paren_count = 0
    
    while i < len(sql_query):
        char = sql_query[i]
        
        # Handle quotes
        if char in ["'", '"'] and (i == 0 or sql_query[i-1] != '\\'):
            current_token += char
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        # Handle parentheses
        elif char == '(' and not in_quotes:
            current_token += char
            paren_count += 1
        elif char == ')' and not in_quotes:
            current_token += char
            paren_count -= 1
        # Handle whitespace
        elif char.isspace() and not in_quotes and paren_count == 0:
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char
        
        i += 1
    
    # Add the final token if any
    if current_token:
        tokens.append(current_token)
    
    return tokens

def find_next_clause(tokens, start_idx):
    """
    Find the index of the next major clause keyword in the token list.
    
    Args:
        tokens (list): List of SQL tokens
        start_idx (int): Starting index to search from
        
    Returns:
        int: Index of the next major clause, or -1 if none found
    """
    clause_keywords = ['JOIN', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'OFFSET', 'UNION', 'EXCEPT', 'INTERSECT']
    
    for i in range(start_idx, len(tokens)):
        token_upper = tokens[i].upper()
        
        # Check exact matches
        if token_upper in clause_keywords:
            return i
        
        # Check for compound keywords (e.g., LEFT JOIN)
        for keyword in ['LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN', 'CROSS JOIN', 'FULL JOIN']:
            if token_upper.endswith(' JOIN') and token_upper in keyword:
                return i
    
    return -1
