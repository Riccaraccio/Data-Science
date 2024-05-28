import pandas as pd
import re

# Load the DataFrame
df = pd.read_excel("satellites.xlsx")

# Display the first few rows of the DataFrame
print(df.head())

# Access columns
print(df["Target"])
print(df["V8"])

# Access rows
print(df.iloc[0]) #by index
print(df.loc[0])  #by label

# Access a specific value
print(df.at[0, 'V8'])
print(df['Target'][0])

# Filter data
filtered_df = df[df['Target'] == 'Anomaly']
print(filtered_df)

# Summarize statistics
print(df.describe())

# Handle missing values
print(df.isnull().sum()) # count missing values
df_cleaned = df.dropna() # drop missing values
df_filled = df.fillna(value={'V8': '0'}) # fill missing values

# Group data
grouped_df = df.groupby('Target').mean()
print(grouped_df)
group_size = df.groupby('Target').size()
print(group_size)

# Add new column
df['new_column'] = df['V8'] + df['V9']
print(df.head())

# Sort data
df_sorted = df.sort_values(by='V8')
print(df_sorted.head())
df_sorted_desc = df.sort_values(by='V8', ascending=False)
print(df_sorted_desc.head())

# Save the DataFrame
df.to_excel("satellites_cleaned.xlsx", index=False)

# iterate over rows
for index, row in df.iterrows():
    print(row['V8'])
    
# iterate over columns
for column in df.columns:
    print(df[column]) 
    
    
    