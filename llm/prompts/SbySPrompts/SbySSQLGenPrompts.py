
# medium generation prompt format
medium_sql_gen_prompt_v0 =  '''
{medium_generation_task}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

{medium_final_instructions}
# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''
# return: final query




medium_query_instructions_2_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities, if the user request all ranking probabilities, don't use it.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
### Important points to consider
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from ALL the tables used, including the ones from the sub-queries.
- Mantain the EXACT COLUMN names as they are in the database, unless the user explicitly asks you to do so in the request, giving you the new name to use. This is crucial for the query to work correctly.
- Mantain the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object', if the query requires it. It is important to be really careful with the use of sub-queries or JOINs, as they can slow down the query.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with the final SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer. If you want to add comments, use the SQL comment format inside the query.
'''



# Advanced generation prompt format
adv_sql_gen_prompt_v0 =  '''
{adv_generation_task}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

{adv_final_instructions}
# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''
# return: final query



adv_query_instructions_2_v3 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities, if the user request all ranking probabilities, don't use it.
- 'probability.classifier_name='lc_classifier'
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, taking only the oids in the 'probability' sub-query with an INNER JOIN inside each 'feature' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the 'UNION' query from step III selecting only the 'oids' that are in the 'INTERSECT' query from step IV
VI. Add the remaining conditions to the final result of step V, using the 'probability' sub-query from step I.
### GENERAL
### Important points to consider
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from ALL the tables used, including the ones from the sub-queries.
- Mantain the EXACT COLUMN names as they are in the database, unless the user explicitly asks you to do so in the request, giving you the new name to use. This is crucial for the query to work correctly.
- Mantain the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.


# If you need to use 2 or 3 tables, try using a sub-query or INNER JOIN over 'probability' TABLE or 'object' TABLE, or an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# Finally, join all the steps in a final query, with the following format: 
```sql [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.
'''

