

# medium decomposition prompt format
medium_decomp_prompt_v0 = '''
{medium_decomp_task}
# General context about the database:
{medium_query_cntx}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}
# Important details about the database required for the query:
{medium_final_instructions}
'''
# return: decomposition steps


medium_decomp_task_v3 = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, and the details of the database schema. This is very important to ensure the query is optimal and accurate.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''


medium_query_instructions_1_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from all the tables that are used in the query.
- Use the exact table and column names as they are in the database. This is crucial for the query to work correctly.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

# If you need to use 2 tables, try using a sub-query or INNER JOIN over 'probability' TABLE or 'object' TABLE, or an INNER JOIN between 'probabbility' and 'object', choosing the best option for the query.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''


# Advanced decomposition prompt format
adv_decomp_prompt_v0 = '''
{adv_decomp_task}
# General context about the database:
{adv_query_cntx}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}
# Important details about the database required for the query:
{adv_final_instructions}
'''
# return: decomposition steps




adv_decomp_task_v3 = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, and the details of the database schema. This is very important to ensure the query is optimal and accurate.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
The request is a very difficult and advanced query, so you will need to use JOINs, INTERSECTs and UNIONs statements, together with Nested queries. It is very important that you give every possible detail in each step, describing the statements and the nested-queries that are required.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''



adv_query_instructions_1_v3 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier'
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

# If you need to use 2 or 3 tables, try using a sub-query or INNER JOIN over 'probability' TABLE or 'object' TABLE, or an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''












gpt4turbo1106_decomposed_prompt_2_v0 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from the ALeRCE astronomy broker database involves several steps. ALeRCE (Automatic Learning for the Rapid Classification of Events) is a system designed to classify large amounts of astronomical data, typically from surveys like the Zwicky Transient Facility (ZTF). To create a detailed and understandable plan, follow these steps:

1. **Understand the Database Schema:**
   - Obtain the database schema, which includes tables, columns, data types, relationships, and constraints.
   - Identify the relevant tables and columns that contain the information you need.

2. **Define the Information Needed:**
   - Clearly specify what information you want to retrieve. For example, you might be interested in transient events, their classifications, light curves, or cross-matches with other catalogs.
   - Determine the level of detail required (e.g., specific time ranges, magnitude limits, or particular sky regions).

3. **Formulate the Query Requirements:**
   - Decide on the selection criteria (e.g., date, magnitude, classification confidence).
   - Determine if you need to join multiple tables and how they are related.
   - Consider if you need to aggregate data (e.g., average magnitudes, count of events).

4. **Design the Query:**
   - Start with the main table that contains the bulk of the information you need.
   - Use `JOIN` clauses to combine related tables based on common keys.
   - Apply `WHERE` clauses to filter the data according to your criteria.
   - Use `GROUP BY` and aggregate functions if necessary.
   - Decide on the sorting order of the results using `ORDER BY`.

5. **Document the Query:**
   - Write comments within the SQL code to explain the purpose of different parts of the query.
   - Create external documentation that describes the query's purpose, the information it retrieves, and any assumptions or limitations.

Here's an example of a simple PostgreSQL query structure based on the steps above:

```sql
-- Retrieve transient events with their classifications and light curves
-- for a specific time range and magnitude limit

SELECT
    e.event_id,
    e.ra,
    e.dec,
    c.classification,
    c.confidence,
    lc.mag,
    lc.time
FROM
    events e
JOIN
    classifications c ON e.event_id = c.event_id
JOIN
    light_curves lc ON e.event_id = lc.event_id
WHERE
    e.time_observed BETWEEN '2023-01-01' AND '2023-01-31'
    AND lc.mag < 20
ORDER BY
    e.time_observed DESC, lc.time ASC;
```

Remember that the actual query will depend on the specific schema and requirements of the ALeRCE database. Always test your queries to ensure they perform as expected and return accurate results. 
'''
gpt4turbo1106_decomposed_prompt_2 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from the ALeRCE astronomy broker database involves several steps. ALeRCE (Automatic Learning for the Rapid Classification of Events) is a system designed to classify large amounts of astronomical data, typically from surveys like the Zwicky Transient Facility (ZTF). To create a detailed and understandable plan, follow these steps:

1. **Understand the Database Schema:**
   - Obtain the database schema, which includes tables, columns, data types, relationships, and constraints.
   - Identify the relevant tables and columns that contain the information you need.

2. **Define the Information Needed:**
   - Clearly specify what information you want to retrieve. For example, you might be interested in transient events, their classifications, light curves, or cross-matches with other catalogs.
   - Determine the level of detail required (e.g., specific time ranges, magnitude limits, or particular sky regions).

3. **Formulate the Query Requirements:**
   - Decide on the selection criteria (e.g., date, magnitude, classification confidence).
   - Determine if you need to join multiple tables and how they are related.
   - Consider if you need to aggregate data (e.g., average magnitudes, count of events).

4. **Design the Query:**
   - Start with the main table that contains the bulk of the information you need.
   - Use `JOIN` clauses to combine related tables based on common keys.
   - Apply `WHERE` clauses to filter the data according to your criteria.
   - Use `GROUP BY` and aggregate functions if necessary.
   - Decide on the sorting order of the results using `ORDER BY`.

5. **Document the Query:**
   - Write comments within the SQL code to explain the purpose of different parts of the query.
   - Create external documentation that describes the query's purpose, the information it retrieves, and any assumptions or limitations.

Remember that the actual query will depend on the specific schema and requirements of the ALeRCE database. Always test your queries to ensure they perform as expected and return accurate results. 
'''


gpt41_0414_decomposed_prompt_v0 = '''
You are tasked with creating a step-by-step decomposition plan for generating a PostgreSQL query to retrieve specific information from the ALeRCE astronomy broker database. The ALeRCE (Automatic Learning for the Rapid Classification of Events) system manages and classifies large volumes of astronomical data, such as transient events from surveys like the Zwicky Transient Facility (ZTF).

**Instructions:**

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

**Example SQL Query Structure:**

```sql
-- Retrieve transient events with their classifications and light curves
-- for a specific time range and magnitude limit

SELECT
    e.event_id,
    e.ra,
    e.dec,
    c.classification,
    c.confidence,
    lc.mag,
    lc.time
FROM
    events e
JOIN
    classifications c ON e.event_id = c.event_id
JOIN
    light_curves lc ON e.event_id = lc.event_id
WHERE
    e.time_observed BETWEEN '2023-01-01' AND '2023-01-31'
    AND lc.mag < 20
ORDER BY
    e.time_observed DESC, lc.time ASC;
```

**Constraints:**
- Tailor the plan and query to the actual ALeRCE database schema and your specific data requirements.
- Ensure all queries are tested for accuracy and performance.
- Provide clear documentation for both the query logic and its implementation.

**Output Format:**
- Present your decomposition plan as a structured list or set of steps, as shown above.
- Ensure all instructions and examples are clear and self-contained.
'''


gpt41_0414_decomposed_prompt = '''
You are tasked with creating a step-by-step decomposition plan for generating a PostgreSQL query to retrieve specific information from the ALeRCE astronomy broker database. The ALeRCE (Automatic Learning for the Rapid Classification of Events) system manages and classifies large volumes of astronomical data, such as transient events from surveys like the Zwicky Transient Facility (ZTF).

**Instructions:**

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

**Output Format:**
- Present your decomposition plan as a structured list or set of steps, as shown above.
- Ensure all instructions and examples are clear and self-contained.
'''


claude_decomposed_prompt_1 = '''
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
- Primary tables: objects, detections, non_detections, probabilities, xmatch_sdss
- Key columns:
  * objects: oid (VARCHAR), firstmjd (DOUBLE), lastmjd (DOUBLE), ra (DOUBLE), dec (DOUBLE)
  * detections: oid (VARCHAR), mjd (DOUBLE), mag (REAL), e_mag (REAL), fid (INTEGER)
  * probabilities: oid (VARCHAR), class_name (VARCHAR), probability (REAL)
  * xmatch_sdss: oid (VARCHAR), sdss_oid (VARCHAR), dist (REAL)
- Foreign key relationships:
  * detections.oid → objects.oid
  * probabilities.oid → objects.oid
  * xmatch_sdss.oid → objects.oid
- Indexing considerations: Ensure indexes exist on objects.lastmjd and detections.mag for performance
</schema_analysis>

[Additional sections would continue in this format...]

## Requirements and Constraints
1. Focus ONLY on planning, not creating the final SQL code
2. Be as specific as possible about table and column names, do not use aliases unless the user specifies them
3. Consider astronomical domain knowledge in your planning
4. Address potential performance issues with large datasets (millions of objects)
5. Include considerations for time-series data handling
6. Reference standard astronomical practices (magnitudes, coordinates, epochs)
7. Consider both selection and exclusion criteria (quality flags, non-detections)

## Evaluation Criteria
Your plan will be evaluated on:
- Completeness (all necessary steps included)
- Astronomical domain awareness
- Logical structure and sequence
- Consideration of edge cases
- Database performance awareness
- Clarity and specificity
'''

claude_decomposed_prompt_2 = '''
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

## Example (Partial)
<query_objective>
Retrieve light curve data for supernova candidates detected in the last 30 days with brightness exceeding 19th magnitude in g-band, including classification probabilities and host galaxy information when available.
</query_objective>

<schema_analysis>
- Primary tables: objects, detections, probabilities, xmatch_sdss
- Key columns:
  * objects: oid (VARCHAR), firstmjd (DOUBLE), lastmjd (DOUBLE), ra/dec (DOUBLE)
  * detections: oid (VARCHAR), mjd (DOUBLE), mag (REAL), fid (INTEGER)
  * probabilities: oid (VARCHAR), class_name (VARCHAR), probability (REAL)
- Foreign key relationships: All tables link to objects.oid
- Indexing: Consider indexes on objects.lastmjd and detections.mag
</schema_analysis>

## Requirements and Constraints
1. Focus ONLY on planning, not creating the final SQL code
2. Be as specific as possible about table and column names, do not use aliases unless the user specifies them
3. Consider astronomical domain knowledge in your planning
4. Address potential performance issues with large datasets (millions of objects)
5. Include considerations for time-series data handling
6. Reference standard astronomical practices (magnitudes, coordinates, epochs)
7. Consider both selection and exclusion criteria (quality flags, non-detections)

## Evaluation Criteria
Your plan will be evaluated on:
- Completeness (all necessary steps included)
- Astronomical domain awareness
- Logical structure and sequence
- Consideration of edge cases
- Database performance awareness
- Clarity and specificity
'''