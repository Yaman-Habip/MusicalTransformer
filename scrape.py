import requests
from artists import rock_artists
from bs4 import BeautifulSoup
import csv

# Generate a list of songs based off of rock artists
def get_songs(artist): # Expects an artist, returns e-chords urls of songs of that artist as a list

  # Creating a BeautifulSoup obejct for a given artists
  response = requests.get("https://www.e-chords.com/" + artist.replace(" ", "-"))
  html_content = response.text
  soup = BeautifulSoup(html_content, 'html.parser')

  # The songs are in <a></a> tags with class 'ta'
  song_links = soup.find_all('a', class_='ta')

  # Adding the href of the a tags to a list and returning it
  return [i["href"] for i in song_links]

# List of the songs we want to scrape
urls = []
for i in rock_artists:
  urls.extend(list(set(get_songs(i))))


def get_chords(url): # Expects an e-chords url, returns chords of that song as a list
  # Creating a BeautifulSoup obejct for a given song
  response = requests.get(url)
  html_content = response.text
  soup = BeautifulSoup(html_content, 'html.parser')

  # The chords are in <u></u> tags
  u_tags = soup.find_all('u')

  # Adding the content of the u tags to a list and returning it
  return [i.text for i in u_tags]

# Writing the chords to a csv file
counter = 0
with open("master_chords.csv", 'w', newline='') as file:
  writer = csv.writer(file)
  for i in urls:
    counter += 1
    print(counter, i)
    writer.writerow(get_chords(i))