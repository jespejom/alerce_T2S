# SQL Line-by-Line Comparison Widget

## Overview

This notebook implements a specialized visualization for comparing SQL queries line by line, focusing on associating similar clauses, conditions, and statements between gold (reference) and predicted SQL queries.

The visualization pipeline:
1. Parses SQL queries into meaningful "lines" (clauses, conditions, etc.)
2. Calculates similarity between lines from predicted and gold queries
3. Associates lines from both queries based on similarity thresholds
4. Visualizes the matching with appropriate highlighting (green for matched, red for unmatched)

## Enhanced SQL Parser

The SQL parser in this widget has been improved to handle various complex SQL structures:

### Line Extraction Improvements

- **Proper Clause Separation**: SQL clauses are now correctly separated with awareness of nested structures and quoted strings
- **Complex Condition Handling**: WHERE and JOIN conditions are properly split with respect to parentheses and function calls
- **Nested Structure Preservation**: Nested expressions and parenthesized groups are kept intact

### Subquery Handling Enhancements

- **Recursive Subquery Parsing**: Subqueries are now parsed recursively to reveal their internal structure
- **Hierarchical Representation**: Subquery lines are properly associated with their parent query components
- **Subquery Type Identification**: Different types of subqueries (SELECT, FROM, WHERE) are correctly identified and labeled

### Key Functions

- `parse_sql_into_lines(sql_query)`: Main parsing function that breaks a SQL query into meaningful lines
- `extract_subqueries(sql_query)`: Detects and extracts subqueries from a SQL query
- `parse_subquery_recursively(subquery_content, subquery_type)`: Parses subqueries to reveal their structure
- `split_sql_clause_safely(clause_content, split_tokens, respect_functions)`: Splits SQL clauses while respecting nested structures

## Helper Functions

The `sql_line_comparison_widget_helpers.py` module contains utility functions:

- `handle_complex_join_conditions(join_condition)`: Splits JOIN conditions into individual parts
- `split_join_clause(join_clause)`: Separates the table part from the ON condition in JOIN clauses
- `extract_on_clauses_from_joins(sql_query)`: Extracts all ON conditions from JOINs in a query
- `preprocess_sql(sql_query)`: Handles comments, quotes, and other special syntax
- `tokenize_sql(sql_query)`: Breaks SQL into tokens while respecting quotes and parentheses

## Usage

To use the widget:

1. Load the evaluation results
2. Initialize the widget with the results dictionary
3. Use the UI controls to navigate through the comparison results

```python
widget = SQLLineComparisonWidget(results_dict, df_train, df_test)
```

## Similarity Methods

The widget offers three similarity methods for comparing SQL lines:

1. **TF-IDF**: Uses TF-IDF vectorization and cosine similarity (default)
2. **Jaccard**: Uses Jaccard similarity (set-based comparison)
3. **Embedding**: Uses neural word embeddings via sentence-transformers
