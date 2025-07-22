


# Advanced generation prompt format
med_decomp_dir_gen_v0 = '''
{medium_generation_task}
{medium_query_cntx}
{medium_final_instructions}
# DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, and the details of the database schema. This is very important to ensure the query is optimal and accurate.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
# Before making the query, use the next step-by-step planification to write the query. Write each step number with the important details of the step.

# Step-by-Step plan

1. **Review the Database Schema**
   - Obtain and examine the ALeRCE database schema, including all tables, columns, data types, relationships, and constraints.
   - Identify which tables and columns are relevant to the information you wish to retrieve.

2. **Specify Data Requirements**
   - Clearly define the exact information you need from the database (e.g., transient events, classifications, light curves, cross-matches).
   - Specify any filters or constraints, such as time ranges, magnitude limits, or sky regions.

3. **Determine Query Logic**
   - List the selection criteria for the data (e.g., date ranges, magnitude thresholds, classification confidence).
   - Identify necessary table joins and the relationships between tables.
   - Decide if data aggregation is required (e.g., averages, counts).

4. **Draft the SQL Query**
   - Begin with the primary table containing the core data.
   - Use appropriate `JOIN` clauses to connect related tables.
   - Apply `WHERE` clauses to filter results according to your criteria.
   - Incorporate `GROUP BY` and aggregate functions if needed.
   - Specify the result ordering using `ORDER BY`.

5. **Document the Query**
   - Add comments within the SQL code to clarify the purpose of each section.
   - Prepare external documentation summarizing the query’s intent, the data it retrieves, and any assumptions or limitations.

**Constraints:**
- Tailor the plan and query to the actual ALeRCE database schema and your specific data requirements.
- Ensure all queries are tested for accuracy and performance.
- Provide clear documentation for both the query logic and its implementation.

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

# Remember to return the SQL code with the following format:
```sql SQL_QUERY_HERE ```
'''




# Advanced generation prompt format
med_decomp_dir_gen_2 = '''
{medium_generation_task}
{medium_query_cntx}
{medium_final_instructions}
# DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, and the details of the database schema. This is very important to ensure the query is optimal and accurate.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
# Before making the query, use the next step-by-step planification to write the query. Write each step number with the important details of the step.

# Step-by-Step plan
Provide a comprehensive planning document with the following sections (use these exact section headers):

<query_objective>
[Clear statement of what information needs to be retrieved and why]
</query_objective>

<schema_analysis>
[Detailed analysis of relevant tables and their relationships]
- Primary tables: [List the main tables needed]
</schema_analysis>

<filtering_criteria>
[Specific conditions to apply]
</filtering_criteria>

<join_strategy>
[Plan for combining tables]
- Join sequence: [Order of table joins]
- Join conditions: [Exact matching columns]
</join_strategy>

<output_structure>
[How results should be organized]
- Column selection: [Final columns to return]
- Sorting criteria: [ORDER BY specifications]
- Pagination approach: [If large result sets expected]
- Result limitations: [LIMIT or other constraints]
</output_structure>

<optimization_notes>
[Performance considerations]
- Query execution plan considerations
- Potential indexing needs
- Subquery vs. join tradeoffs
</optimization_notes>

## Example Output
Here's an example of what a good response might look like:

<query_objective>
Create a plan to retrieve light curve data for all supernova candidates detected in the last 30 days with brightness exceeding 19th magnitude in the g-band, including their classification probabilities and host galaxy information when available.
</query_objective>

<schema_analysis>
- Primary tables: objects, detection, probability
- Key columns:
  * objects: oid (VARCHAR), firstmjd (DOUBLE), lastmjd (DOUBLE), ra (DOUBLE), dec (DOUBLE)
  * detection: oid (VARCHAR), mjd (DOUBLE), mag (REAL), e_mag (REAL), fid (INTEGER)
  * probability: oid (VARCHAR), class_name (VARCHAR), probability (REAL)
- Foreign key relationships:
  * detection.oid → objects.oid
  * probability.oid → objects.oid
- Indexing considerations: Ensure indexes exist on objects.lastmjd and detection.mag for performance
</schema_analysis>

[Additional sections would continue in this format...]

```sql SQL_QUERY_HERE ```

## Requirements and Constraints
1. Be as specific as possible about table and column names, do not use aliases unless the user specifies them
2. Consider astronomical domain knowledge in your planning
3. Be careful with the use of sub-queries or JOINs, as they can return different data from what the user is asking for.
4. Do not overuse sub-queries or JOINs, as they can slow down the query.
5. Do not overcomplicate the SQL code, go straight to the point.

# Remember to return the SQL code with the following format:
```sql SQL_QUERY_HERE ```
'''



# final instructions with the most important details
med_query_instructions_sbscot_v0 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from all the tables that are used in the query.
- Use the exact table and column names as they are in the database. This is crucial for the query to work correctly.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
- If you need to use 2 tables, try using a sub-query or INNER JOIN over 'probability' TABLE or 'object' TABLE, or an INNER JOIN between 'probability' and 'object', choosing the best option for the query.

### Important points to consider
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from ALL the tables used, including the ones from the sub-queries.
- Mantain the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
- Mantain the EXACT COLUMN names as they are in the database, unless the user EXPLICITLY ASKS you to change the column names in the request, giving you the new name to use. This is crucial for the query to work correctly.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object', if the query requires it. It is important to be really careful with the use of sub-queries or JOINs, as they can slow down the query.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# At the end of your answer, give the SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
'''










# Advanced generation prompt format
adv_decomp_dir_gen_v0 = '''
{adv_generation_task}
{adv_query_cntx}
{adv_final_instructions}

# DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, and the details of the database schema. This is very important to ensure the query is optimal and accurate.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
# Before making the query, use the next step-by-step planification to write the query. Write each step number with the important details of the step.

# Step-by-Step plan
Provide a comprehensive planning document with the following sections (use these exact section headers):

<query_objective>
[Clear statement of what information needs to be retrieved and why]
</query_objective>

<schema_analysis>
[Detailed analysis of relevant tables and their relationships]
- Primary tables: [List the main tables needed]
- Key columns: [List essential columns with their data types]
- Foreign key relationships: [Describe how tables are connected]
- Indexing considerations: [Note any performance considerations]
</schema_analysis>

<filtering_criteria>
[Specific conditions to apply]
- Time range: [Specify time constraints]
- Spatial constraints: [Any RA/Dec regions]
- Magnitude/flux limits: [Brightness thresholds]
- Classification constraints: [Event types of interest]
- Data quality flags: [Required quality indicators]
</filtering_criteria>

<join_strategy>
[Plan for combining tables]
- Join sequence: [Order of table joins]
- Join conditions: [Exact matching columns]
- Join types: [INNER, LEFT, etc. with justification]
- Potential performance issues: [Identify possible bottlenecks]
</join_strategy>

<aggregation_plan>
[If needed, how data should be grouped]
- Grouping columns: [What to group by]
- Aggregate functions: [COUNT, AVG, etc.]
- Having conditions: [Post-aggregation filters]
</aggregation_plan>

<output_structure>
[How results should be organized]
- Column selection: [Final columns to return]
- Sorting criteria: [ORDER BY specifications]
- Pagination approach: [If large result sets expected]
- Result limitations: [LIMIT or other constraints]
</output_structure>

<optimization_notes>
[Performance considerations]
- Query execution plan considerations
- Potential indexing needs
- Subquery vs. join tradeoffs
- Common Table Expressions (CTEs) if helpful
</optimization_notes>

## Example Output
Here's an example of what a good response might look like:

<query_objective>
Create a plan to retrieve light curve data for all supernova candidates detected in the last 30 days with brightness exceeding 19th magnitude in the g-band, including their classification probabilities and host galaxy information when available.
</query_objective>

<schema_analysis>
- Primary tables: objects, detections, non_detections, probability, xmatch_sdss
- Key columns:
  * objects: oid (VARCHAR), firstmjd (DOUBLE), lastmjd (DOUBLE), ra (DOUBLE), dec (DOUBLE)
  * detections: oid (VARCHAR), mjd (DOUBLE), mag (REAL), e_mag (REAL), fid (INTEGER)
  * probability: oid (VARCHAR), class_name (VARCHAR), probability (REAL)
  * xmatch_sdss: oid (VARCHAR), sdss_oid (VARCHAR), dist (REAL)
- Foreign key relationships:
  * detections.oid → objects.oid
  * probability.oid → objects.oid
  * xmatch_sdss.oid → objects.oid
- Indexing considerations: Ensure indexes exist on objects.lastmjd and detections.mag for performance
</schema_analysis>

[Additional sections would continue in this format...]

```sql SQL_QUERY_HERE ```

## Requirements and Constraints
1. Be as specific as possible about table and column names, do not use aliases unless the user specifies them
2. Consider astronomical domain knowledge in your planning
3. Consider both selection and exclusion criteria (quality flags, non-detections)

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

# Remember to return the SQL code with the following format:
```sql SQL_QUERY_HERE ```
'''




# final instructions with the most important details
adv_query_instructions_sbscot_v0 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, taking only the oids in the 'probability' sub-query with an INNER JOIN inside each 'feature' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the 'UNION' query from step III selecting only the 'oids' that are in the 'INTERSECT' query from step IV
VI. Add the remaining conditions to the final result of step V, using the 'probability' sub-query from step I.
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Use the exact table and column names as they are in the database. This is crucial for the query to work correctly.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
- If you need to use 2 or 3 tables, try using a sub-query or INNER JOIN over 'probability' TABLE or 'object' TABLE, or an INNER JOIN between 'probability' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).

### Important points to consider
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from ALL the tables used, including the ones from the sub-queries.
- Mantain the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
- Mantain the EXACT COLUMN names as they are in the database, unless the user EXPLICITLY ASKS you to change the column names in the request, giving you the new name to use. This is crucial for the query to work correctly.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object', if the query requires it. It is important to be really careful with the use of sub-queries or JOINs, as they can slow down the query.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# At the end of your answer, give the SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
'''



sbs_simple_cot_v0 = '''
{general_task}

Database Schema:
<db_schema>
{db_schema}
</db_schema>

{general_context}

{final_instructions}

Now, analyze the request and generate an appropriate PostgreSQL query:

<query_planning>
1. Summarize the user's request:
   [Provide a brief summary of what the user is asking for]

2. Interpret the main objectives of the query:
   [List the key goals the query needs to achieve]

3. Identify relevant tables and columns:
   [List the tables and key columns needed]

4. List specific columns required:
   [Enumerate the exact columns that should be included in the SELECT statement]

5. Determine necessary conditions and filters:
   [Outline the WHERE clauses and any JOINs required]

6. Consider spatial query requirements:
   [Evaluate if Q3C functions are needed and how to implement them]

7. Evaluate need for aggregations or groupings:
   [Determine if any GROUP BY, HAVING, or aggregate functions are necessary]

8. Optimize for simplicity and performance:
   [Ensure the query is as simple as possible while meeting all requirements]

9. Verify adherence to default conditions and naming conventions:
   [Check that default conditions are applied and names are correct]

10. Consider potential edge cases or error handling:
    [Think about possible issues and how to address them in the query]

11. Final query structure:
    [Outline the overall structure of the SQL query]
</query_planning>

'''

sbs_adv_cot_v0 = '''
{adv_generation_task}

Database Schema:
<db_schema>
{db_schema}
</db_schema>

{adv_query_cntx}

{adv_final_instructions}

Now, analyze the request and generate an appropriate PostgreSQL query:

<query_planning>
1. Summarize the user's request:
   [Provide a brief summary of what the user is asking for]

2. Interpret the main objectives of the query:
   [List the key goals the query needs to achieve]

3. Identify relevant tables and columns:
   [List the tables and key columns needed]

4. List specific columns required:
   [Enumerate the exact columns that should be included in the SELECT statement]

5. Determine necessary conditions and filters:
   [Outline the WHERE clauses and any JOINs required]

6. Consider spatial query requirements:
   [Evaluate if Q3C functions are needed and how to implement them]

7. Evaluate need for aggregations or groupings:
   [Determine if any GROUP BY, HAVING, or aggregate functions are necessary]

8. Optimize for simplicity and performance:
   [Ensure the query is as simple as possible while meeting all requirements]

9. Verify adherence to default conditions and naming conventions:
   [Check that default conditions are applied and names are correct]

10. Consider potential edge cases or error handling:
    [Think about possible issues and how to address them in the query]

11. Final query structure:
    [Outline the overall structure of the SQL query]
</query_planning>

'''

sbs_medium_cot_v0 = '''
{medium_generation_task}

Database Schema:
<db_schema>
{db_schema}
</db_schema>

{medium_query_cntx}

{medium_final_instructions}

Now, analyze the request and generate an appropriate PostgreSQL query:

<query_planning>
1. Summarize the user's request:
   [Provide a brief summary of what the user is asking for]

2. Interpret the main objectives of the query:
   [List the key goals the query needs to achieve]

3. Identify relevant tables and columns:
   [List the tables and key columns needed]

4. List specific columns required:
   [Enumerate the exact columns that should be included in the SELECT statement]

5. Determine necessary conditions and filters:
   [Outline the WHERE clauses and any JOINs required]

6. Consider spatial query requirements:
   [Evaluate if Q3C functions are needed and how to implement them]

7. Evaluate need for aggregations or groupings:
   [Determine if any GROUP BY, HAVING, or aggregate functions are necessary]

8. Optimize for simplicity and performance:
   [Ensure the query is as simple as possible while meeting all requirements]

9. Verify adherence to default conditions and naming conventions:
   [Check that default conditions are applied and names are correct]

10. Consider potential edge cases or error handling:
    [Think about possible issues and how to address them in the query]

11. Final query structure:
    [Outline the overall structure of the SQL query]
</query_planning>

'''
# Based on the above analysis, here is the PostgreSQL query:

sbs_task_cot_v0='''
You are an AI assistant specialized in generating PostgreSQL queries for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database. Your task is to create accurate and efficient queries based on user requests while adhering to specific guidelines and database structure.
'''
sbs_context_cot_v0='''
Context:

1. ALeRCE Pipeline Details:
- Stamp Classifier ('stamp_classifier'): Classifies all alerts related to new objects.
- Light Curve Classifier ('lc_classifier'): A balanced hierarchical random forest classifier with four models and 15 classes.
- Hierarchical classifiers:
  - 'lc_classifier_top': [Periodic, Stochastic, Transient]
  - 'lc_classifier_periodic': [LPV, E, DSCT, RRL, CEP, Periodic-Other]
  - 'lc_classifier_transient': [SNIa, SNIbc, SNII, SLSN]
  - 'lc_classifier_stochastic': [QSO, AGN, Blazar, CV/Nova, YSO]

2. Q3C Extension Functions:
- q3c_dist(ra1, dec1, ra2, dec2): Returns distance in degrees between two points.
- q3c_join(ra1, dec1, ra2, dec2, radius): Returns true if (ra1, dec1) is within radius of (ra2, dec2).
- q3c_ellipse_join(ra1, dec1, ra2, dec2, major, ratio, pa): Like q3c_join, but for elliptical regions.
- q3c_radial_query(ra, dec, center_ra, center_dec, radius): Returns true if (ra, dec) is within radius of (center_ra, center_dec).
- q3c_ellipse_query(ra, dec, center_ra, center_dec, maj_ax, axis_ratio, PA): Returns true if (ra, dec) is within specified ellipse.
- q3c_poly_query(ra, dec, poly): Returns true if (ra, dec) is within specified spherical polygon.
'''

sbs_instructions_cot_v0='''
Instructions:

1. Analyze the user's request carefully.
2. Identify the relevant tables and columns needed for the query.
3. Apply default conditions unless explicitly overridden:
   - For 'probability' table: Use 'probability.ranking = 1' and 'probability.classifier_name = 'lc_classifier''.
4. Use INNER JOIN or sub-queries when necessary, but be cautious about query performance.
5. Maintain exact column and class names as in the database, unless explicitly requested otherwise.
6. Use Q3C functions for spatial queries when appropriate.
7. If no specific columns are requested, use "SELECT *" from all relevant tables.
8. Add PostgreSQL comments to explain complex parts of the query if necessary.
9. Ensure the query is simple and concise while meeting all requirements.

'''