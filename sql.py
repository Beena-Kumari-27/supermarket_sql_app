import sqlite3
##connect to sqlite
connection=sqlite3.connect("supermarket.db")
##create a cursor obj to insert record ,create table,retrieve
cursor=connection.cursor()
##create the table
table_info = """
CREATE TABLE IF NOT EXISTS Stock (
    PNAME VARCHAR(25),
    BRAND VARCHAR(25),
    QUANTITY VARCHAR(25),
    PRICE INT,
    EXPIRYDATE VARCHAR(20)
);
"""

# Execute the CREATE TABLE command
cursor.execute(table_info)

##inserting records
cursor.execute('''Insert Into Stock values('Milk','Amul','29','40','15-5-2025')''')
cursor.execute('''Insert Into Stock values('Bread','Arun','40','55','27-5-2025')''')
cursor.execute('''Insert Into Stock values('Eggs','Sneha','50','6','30-5-2025')''')
cursor.execute('''Insert Into Stock values('Biscuits','Parle','23','20','11-6-2025')''')
cursor.execute('''Insert Into Stock values('Chocolate','Cadbury','29','40','10-5-2025')''')
##display all the records
print("The inserted records are")
data=cursor.execute('''Select * From Stock''')
for row in data:
    print(row)

##close the connection
connection.commit()
connection.close()