
def save_report_to_file(report_content, filename):
    """
    Save a report to a text file.

    Args:
        report_content (str): The content of the report to be saved.
        filename (str): The name of the text file to save the report to.

    Returns:
        bool: True if the report was successfully saved, False otherwise.
    """
    try:
        # Open the file in write mode and create it if it doesn't exist
        with open(filename, 'w') as file:
            # Write the report content to the file
            file.write(report_content)
        
        print(f'Report saved to {filename} successfully.')
        return True
    except Exception as e:
        print(f'Error saving the report: {str(e)}')
        return False