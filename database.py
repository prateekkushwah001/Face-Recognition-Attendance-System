# database.py
import sqlite3
from datetime import date, datetime
import csv
import os

def initialize_db():
    """Initializes the database and creates the attendance table if it doesn't exist."""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Check if table exists and has correct structure
    cursor.execute("PRAGMA table_info(attendance)")
    columns = cursor.fetchall()
    
    if not columns:
        # Table doesn't exist, create it
        cursor.execute('''
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(name, attendance_date)
            )
        ''')
    else:
        # Table exists, check if it has the right structure
        column_names = [col[1] for col in columns]
        expected_columns = ['id', 'name', 'attendance_date', 'timestamp']
        
        if len(column_names) > 4:
            # Table has extra columns from previous versions, recreate it
            cursor.execute('DROP TABLE attendance')
            cursor.execute('''
                CREATE TABLE attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    attendance_date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(name, attendance_date)
                )
            ''')
    
    conn.commit()
    conn.close()

def mark_attendance(name):
    """Marks attendance for a given name in the database."""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    try:
        cursor.execute("INSERT INTO attendance (name, attendance_date, timestamp) VALUES (?, ?, ?)",
                       (name, today, timestamp))
        conn.commit()
        export_to_csv()
        conn.close()
        return True, f"Attendance marked successfully for {name}"
    except sqlite3.IntegrityError as e:
        # This handles cases where attendance for the person on the same day already exists
        conn.close()
        return False, f"Attendance already marked for {name} today"
    except Exception as e:
        conn.close()
        return False, f"Database error: {e}"

def has_attended_today(name):
    """Checks if a person has already been marked present today."""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")
    cursor.execute("SELECT 1 FROM attendance WHERE name = ? AND attendance_date = ?", (name, today))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def export_to_csv():
    """Exports today's attendance data to a CSV file."""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")
    
    # Define CSV filename with today's date
    folder_name = "Attendance_Reports"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_file_path = os.path.join(folder_name, f"Attendance_{today}.csv")

    cursor.execute("""
        SELECT attendance_date, timestamp, name 
        FROM attendance 
        WHERE attendance_date = ? 
        ORDER BY timestamp
    """, (today,))
    records = cursor.fetchall()
    
    # Always create the file, even if no records exist
    headers = ["Date", "Time", "Name"]
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        if records:
            writer.writerows(records)
        else:
            # Add a placeholder row for empty days
            writer.writerow([today, "No records", "No attendance yet"])
    
    if records:
        pass  # Records exported successfully
    else:
        pass  # Empty file created for today
    
    conn.close()
    return csv_file_path

def get_daily_attendance_summary():
    """Get a summary of today's attendance"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")
    
    # Get total attendees today
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date = ?", (today,))
    total_attendees = cursor.fetchone()[0]
    
    # Get detailed attendance info
    cursor.execute("""
        SELECT name, timestamp 
        FROM attendance 
        WHERE attendance_date = ? 
        ORDER BY timestamp
    """, (today,))
    attendees = cursor.fetchall()
    
    conn.close()
    
    return {
        'date': today,
        'total_attendees': total_attendees,
        'attendees': attendees
    }