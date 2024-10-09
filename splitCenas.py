def separate_genres(input_file, comedy_file, drama_file, romance_file):
    # Open the input file and the output files for writing
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(comedy_file, 'w', encoding='utf-8') as comedy_out, \
         open(drama_file, 'w', encoding='utf-8') as drama_out, \
         open(romance_file, 'w', encoding='utf-8') as romance_out:
        
        # Process each line in the input file
        for line in infile:
            # Split the line by tab to extract the genre
            parts = line.strip().split('\t')
            
            if len(parts) < 5:
                continue  # If line doesn't have enough fields, skip it
            
            genre = parts[2]  # Assuming genre is the third field in the line
            
            # Write to the appropriate file based on the genre
            if genre.lower() == 'comedy':
                comedy_out.write(line)
            elif genre.lower() == 'drama':
                drama_out.write(line)
            elif genre.lower() == 'romance':
                romance_out.write(line)

# File paths
input_file = 'train.txt'     # Input file with all data
comedy_file = 'comedy.txt'    # Output file for comedy genre
drama_file = 'drama.txt'      # Output file for drama genre
romance_file = 'romance.txt'  # Output file for romance genre

# Run the function
separate_genres(input_file, comedy_file, drama_file, romance_file)

print(f"Lines with genre 'comedy' copied to {comedy_file}")
print(f"Lines with genre 'drama' copied to {drama_file}")
print(f"Lines with genre 'romance' copied to {romance_file}")
