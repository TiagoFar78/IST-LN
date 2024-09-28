import re

regex_pattern = '(.*)\t(.*)\t(.*)\t(.*)\t(.*)'

def extract_movie_info(line):
    match = re.match(regex_pattern, line)
    if match:
        title, movie_from, genre, director, plot = match.groups()
        return title, movie_from, genre, director, plot

file_path = 'train.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        
        title, movie_from, genre, director, plot = extract_movie_info(line)
        print(f"Title: {title}, From: {movie_from}, Genre: {genre}, Director: {director}, Plot size: {len(plot)}")
        
