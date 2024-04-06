# import sqlite3
# from datetime import datetime
#
# conn = sqlite3.connect("modified_year.db")
#
# cursor = conn.execute(
#     "SELECT ID, Date FROM AssetsCategories WHERE Date NOT LIKE '%-12-31'"
# )
#
# x = [row for row in cursor]
#
# print(x)
# print(len(x))
#
# for row in x:
#     cur = conn.cursor()
#     cur.execute(
#         f"UPDATE AssetsCategories SET Date = ? WHERE ID = ?", (datetime.strptime(row[1][:4] + '-12-31', '%Y-%m-%d').date(), row[0])
#     )
#     conn.commit()