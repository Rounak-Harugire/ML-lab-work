from datetime import datetime

# Getting Current Date and Time
now = datetime.now()
print("Current Date Is:\n", now)

# Creating Date_Time Object
date_obj = datetime(2004, 1, 1, 6, 15)
print("Birth_Date Is:\n", date_obj)

# Convert String to Datetime
date_str = '2004-01-01 6:15:00'
date_convert = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
print("Converted String TO DateTime Is:\n", date_convert)

# Convert Datetime to String
formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
print("Converted DateTime TO String Is:\n", formatted_date)

# Difference Between Two Dates
time_diff = datetime.now() - date_obj
print("Difference Between Dates Is:\n", time_diff)
