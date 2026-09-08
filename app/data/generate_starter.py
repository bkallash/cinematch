import json
from pathlib import Path

# Script to generate a rich starter catalog of famous movies and TV series
titles = [
    # Sci-Fi / Thriller / Mind-Bending
    {
        "tmdb_id": 157336, "media_type": "movie", "title": "Interstellar", "release_year": 2014,
        "genres": ["Adventure", "Drama", "Science Fiction"], "director_or_creator": "Christopher Nolan",
        "cast_top": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Michael Caine"],
        "poster_path": "/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg", "vote_average": 8.4, "vote_count": 34000, "popularity": 180.0,
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage."
    },
    {
        "tmdb_id": 27205, "media_type": "movie", "title": "Inception", "release_year": 2010,
        "genres": ["Action", "Science Fiction", "Adventure"], "director_or_creator": "Christopher Nolan",
        "cast_top": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Tom Hardy"],
        "poster_path": "/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg", "vote_average": 8.4, "vote_count": 36000, "popularity": 150.0,
        "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life as payment for a task considered to be impossible: inception."
    },
    {
        "tmdb_id": 603, "media_type": "movie", "title": "The Matrix", "release_year": 1999,
        "genres": ["Action", "Science Fiction"], "director_or_creator": "Lana Wachowski",
        "cast_top": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving"],
        "poster_path": "/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg", "vote_average": 8.2, "vote_count": 25000, "popularity": 110.0,
        "overview": "Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth."
    },
    {
        "tmdb_id": 335984, "media_type": "movie", "title": "Blade Runner 2049", "release_year": 2017,
        "genres": ["Science Fiction", "Drama"], "director_or_creator": "Denis Villeneuve",
        "cast_top": ["Ryan Gosling", "Harrison Ford", "Ana de Armas", "Sylvia Hoeks"],
        "poster_path": "/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg", "vote_average": 7.6, "vote_count": 13000, "popularity": 95.0,
        "overview": "Thirty years after the events of the first film, a new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos."
    },
    {
        "tmdb_id": 438631, "media_type": "movie", "title": "Dune", "release_year": 2021,
        "genres": ["Science Fiction", "Adventure"], "director_or_creator": "Denis Villeneuve",
        "cast_top": ["Timothée Chalamet", "Rebecca Ferguson", "Oscar Isaac", "Zendaya"],
        "poster_path": "/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg", "vote_average": 7.8, "vote_count": 11500, "popularity": 140.0,
        "overview": "Paul Atreides, a brilliant and gifted young man born into a great destiny beyond his understanding, must travel to the most dangerous planet in the universe to ensure the future of his family and his people."
    },
    {
        "tmdb_id": 693134, "media_type": "movie", "title": "Dune: Part Two", "release_year": 2024,
        "genres": ["Science Fiction", "Adventure"], "director_or_creator": "Denis Villeneuve",
        "cast_top": ["Timothée Chalamet", "Zendaya", "Rebecca Ferguson", "Javier Bardem"],
        "poster_path": "/6izwz7rsy95ARzTR3poZ8H6c5pp.jpg", "vote_average": 8.2, "vote_count": 5500, "popularity": 190.0,
        "overview": "Follow the mythic journey of Paul Atreides as he unites with Chani and the Fremen while on a path of revenge against the conspirators who destroyed his family."
    },
    {
        "tmdb_id": 329865, "media_type": "movie", "title": "Arrival", "release_year": 2016,
        "genres": ["Drama", "Science Fiction", "Mystery"], "director_or_creator": "Denis Villeneuve",
        "cast_top": ["Amy Adams", "Jeremy Renner", "Forest Whitaker", "Michael Stuhlbarg"],
        "poster_path": "/pEzNVQfdzYDzVK0XqxERIw2x2se.jpg", "vote_average": 7.6, "vote_count": 17000, "popularity": 75.0,
        "overview": "Taking place after alien crafts land around the world, an expert linguist is recruited by the military to determine whether they come in peace or are a threat."
    },
    # Romance / Drama / Vibe
    {
        "tmdb_id": 313369, "media_type": "movie", "title": "La La Land", "release_year": 2016,
        "genres": ["Comedy", "Drama", "Romance", "Music"], "director_or_creator": "Damien Chazelle",
        "cast_top": ["Ryan Gosling", "Emma Stone", "John Legend", "Rosemarie DeWitt"],
        "poster_path": "/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg", "vote_average": 7.9, "vote_count": 16000, "popularity": 85.0,
        "overview": "Mia, an aspiring actress, and Sebastian, a dedicated jazz musician, are struggling to make ends meet in a city known for crushing hopes and breaking hearts. Set in modern-day Los Angeles, this original musical about everyday life explores the joy and pain of pursuing your dreams."
    },
    {
        "tmdb_id": 38, "media_type": "movie", "title": "Eternal Sunshine of the Spotless Mind", "release_year": 2004,
        "genres": ["Science Fiction", "Drama", "Romance"], "director_or_creator": "Michel Gondry",
        "cast_top": ["Jim Carrey", "Kate Winslet", "Kirsten Dunst", "Mark Ruffalo"],
        "poster_path": "/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg", "vote_average": 8.1, "vote_count": 14500, "popularity": 65.0,
        "overview": "Joel Barish, heartbroken that his girlfriend Clementine underwent a procedure to erase him from her memory, decides to do the same. However, as he watches his memories of her fade away, he realizes he still loves her, and may be too late to correct his mistake."
    },
    {
        "tmdb_id": 76, "media_type": "movie", "title": "Before Sunrise", "release_year": 1995,
        "genres": ["Drama", "Romance"], "director_or_creator": "Richard Linklater",
        "cast_top": ["Ethan Hawke", "Julie Delpy", "Andrea Eckert", "Hanno Pöschl"],
        "poster_path": "/kf1Jb1c2JAOqjuzA3H4oDM263uB.jpg", "vote_average": 8.0, "vote_count": 4200, "popularity": 45.0,
        "overview": "On a train from Budapest to Vienna, Jesse meets Céline, a student returning to Paris. After chatting, Jesse convinces Céline to get off the train with him in Vienna so they can spend the night walking through the city talking about love and life before he catches a flight home."
    },
    {
        "tmdb_id": 80, "media_type": "movie", "title": "Before Sunset", "release_year": 2004,
        "genres": ["Drama", "Romance"], "director_or_creator": "Richard Linklater",
        "cast_top": ["Ethan Hawke", "Julie Delpy", "Vernon Dobtcheff", "Louise Lemoine Torrès"],
        "poster_path": "/4sW5XH9ZfYXpvFzev00S1IGAEbg.jpg", "vote_average": 7.9, "vote_count": 3400, "popularity": 38.0,
        "overview": "Nine years after their brief encounter in Vienna, Jesse and Céline meet again in Paris during Jesse's book tour. With only a few hours before Jesse's plane departs, they roam the streets of Paris examining their past choices and lingering chemistry."
    },
    {
        "tmdb_id": 666277, "media_type": "movie", "title": "Past Lives", "release_year": 2023,
        "genres": ["Drama", "Romance"], "director_or_creator": "Celine Song",
        "cast_top": ["Greta Lee", "Teo Yoo", "John Magaro", "Moon Seung-ah"],
        "poster_path": "/k3waqVXSnvCZWfJYNtdamTgTtTA.jpg", "vote_average": 7.8, "vote_count": 1800, "popularity": 60.0,
        "overview": "Nora and Hae Sung, two deeply connected childhood friends, are wrested apart after Nora's family emigrates from South Korea. Two decades later, they are reunited in New York for one fateful week as they confront notions of destiny, love, and the choices that make a life."
    },
    {
        "tmdb_id": 152601, "media_type": "movie", "title": "Her", "release_year": 2013,
        "genres": ["Romance", "Science Fiction", "Drama"], "director_or_creator": "Spike Jonze",
        "cast_top": ["Joaquin Phoenix", "Scarlett Johansson", "Amy Adams", "Rooney Mara"],
        "poster_path": "/eCOtqtfvn7mxGl6nfmq4b1exJRc.jpg", "vote_average": 7.9, "vote_count": 13900, "popularity": 55.0,
        "overview": "In the near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need."
    },
    {
        "tmdb_id": 597, "media_type": "movie", "title": "Titanic", "release_year": 1997,
        "genres": ["Drama", "Romance"], "director_or_creator": "James Cameron",
        "cast_top": ["Leonardo DiCaprio", "Kate Winslet", "Billy Zane", "Kathy Bates"],
        "poster_path": "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg", "vote_average": 7.9, "vote_count": 24800, "popularity": 135.0,
        "overview": "101-year-old Rose DeWitt Bukater tells the story of her life aboard the Titanic, 84 years later. A young Rose boards the ship with her mother and fiance. Meanwhile, Jack Dawson and Fabrizio De Rossi win third-class tickets aboard the ship."
    },
    {
        "tmdb_id": 4348, "media_type": "movie", "title": "Pride & Prejudice", "release_year": 2005,
        "genres": ["Drama", "Romance"], "director_or_creator": "Joe Wright",
        "cast_top": ["Keira Knightley", "Matthew Macfadyen", "Brenda Blethyn", "Donald Sutherland"],
        "poster_path": "/o8UhmEbWPHmTUxP0lMuCoqNkbB3.jpg", "vote_average": 8.1, "vote_count": 8200, "popularity": 62.0,
        "overview": "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy. But Mr. Darcy reluctantly finds himself falling in love with a woman beneath his class."
    },
    {
        "tmdb_id": 19913, "media_type": "movie", "title": "(500) Days of Summer", "release_year": 2009,
        "genres": ["Comedy", "Drama", "Romance"], "director_or_creator": "Marc Webb",
        "cast_top": ["Joseph Gordon-Levitt", "Zooey Deschanel", "Geoffrey Arend", "Chloë Grace Moretz"],
        "poster_path": "/qXAuQ9hF30sQRsXf40OfRVl0MJZ.jpg", "vote_average": 7.3, "vote_count": 9800, "popularity": 45.0,
        "overview": "Tom, greeting-card writer and hopeless romantic, is caught completely off-guard when his girlfriend, Summer, suddenly dumps him. He reflects on their 500 days together to try to figure out where their love affair went sour."
    },
    # Holiday / Christmas Vibes
    {
        "tmdb_id": 1585, "media_type": "movie", "title": "It's a Wonderful Life", "release_year": 1946,
        "genres": ["Drama", "Family", "Fantasy"], "director_or_creator": "Frank Capra",
        "cast_top": ["James Stewart", "Donna Reed", "Lionel Barrymore", "Thomas Mitchell"],
        "poster_path": "/bSqt9rhDZx1Q7UZ86dBPKdNomp2.jpg", "vote_average": 8.3, "vote_count": 4200, "popularity": 50.0,
        "overview": "A holiday favourite for generations... George Bailey has spent his entire life giving to the people of Bedford Falls. All that prevents rich skinflint Mr. Potter from taking over the town is George's modest building and loan company. But on Christmas Eve the business's $8,000 is lost and George's troubles begin."
    },
    {
        "tmdb_id": 771, "media_type": "movie", "title": "Home Alone", "release_year": 1990,
        "genres": ["Comedy", "Family"], "director_or_creator": "Chris Columbus",
        "cast_top": ["Macaulay Culkin", "Joe Pesci", "Daniel Stern", "John Heard"],
        "poster_path": "/onTSipZ8R3bliBdKfPtsDuHTdlL.jpg", "vote_average": 7.4, "vote_count": 11500, "popularity": 90.0,
        "overview": "Eight-year-old Kevin McCallister makes the most of the situation after his family unwittingly leaves him behind when they go on Christmas vacation. But when a pair of bungling burglars try to break in, Kevin must defend his home."
    },
    {
        "tmdb_id": 562, "media_type": "movie", "title": "Die Hard", "release_year": 1988,
        "genres": ["Action", "Thriller"], "director_or_creator": "John McTiernan",
        "cast_top": ["Bruce Willis", "Alan Rickman", "Bonnie Bedelia", "Reginald VelJohnson"],
        "poster_path": "/7Bjd8kfmDSOzpmhySpEhkUyK2oH.jpg", "vote_average": 7.8, "vote_count": 11000, "popularity": 70.0,
        "overview": "NYPD cop John McClane's plan to reconcile with his estranged wife is thrown into serious jeopardy when, after he arrives at her office's Christmas Eve party, the building is overtaken by a group of terrorists."
    },
    {
        "tmdb_id": 5255, "media_type": "movie", "title": "The Polar Express", "release_year": 2004,
        "genres": ["Animation", "Family", "Adventure", "Fantasy"], "director_or_creator": "Robert Zemeckis",
        "cast_top": ["Tom Hanks", "Leslie Zemeckis", "Eddie Deezen", "Nona Gaye"],
        "poster_path": "/eOoCzH0MqeGr2taUZO4SwG416PF.jpg", "vote_average": 6.7, "vote_count": 6200, "popularity": 58.0,
        "overview": "When a doubting young boy takes an extraordinary train ride to the North Pole, he embarks on a journey of self-discovery that shows him that the wonder of life never fades for those who believe."
    },
    {
        "tmdb_id": 5825, "media_type": "movie", "title": "National Lampoon's Christmas Vacation", "release_year": 1989,
        "genres": ["Comedy"], "director_or_creator": "Jeremiah S. Chechik",
        "cast_top": ["Chevy Chase", "Beverly D'Angelo", "Juliette Lewis", "Johnny Galecki"],
        "poster_path": "/oat42hUw8XzKYUmfy0YLAxYd484.jpg", "vote_average": 7.3, "vote_count": 2700, "popularity": 48.0,
        "overview": "It's Christmas time and the Griswolds are preparing for a family seasonal celebration, but things never run smoothly for Clark, his wife Ellen and their two kids."
    },
    # Psychological Thriller / Crime / Drama
    {
        "tmdb_id": 550, "media_type": "movie", "title": "Fight Club", "release_year": 1999,
        "genres": ["Drama"], "director_or_creator": "David Fincher",
        "cast_top": ["Brad Pitt", "Edward Norton", "Helena Bonham Carter", "Meat Loaf"],
        "poster_path": "/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg", "vote_average": 8.4, "vote_count": 29000, "popularity": 110.0,
        "overview": "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy. Their concept catches on, with underground 'fight clubs' forming in every town, until an eccentric gets in the way and ignites an out-of-control spiral toward oblivion."
    },
    {
        "tmdb_id": 807, "media_type": "movie", "title": "Se7en", "release_year": 1995,
        "genres": ["Crime", "Mystery", "Thriller"], "director_or_creator": "David Fincher",
        "cast_top": ["Brad Pitt", "Morgan Freeman", "Gwyneth Paltrow", "Kevin Spacey"],
        "poster_path": "/191nKfP0ehp3uIvWqgPbFmI4lv9.jpg", "vote_average": 8.4, "vote_count": 20800, "popularity": 92.0,
        "overview": "Two homicide detectives are on a desperate hunt for a serial killer whose crimes are based on the 'seven deadly sins' in this dark and haunting film that takes viewers from the tortured remains of one victim to the next."
    },
    {
        "tmdb_id": 278, "media_type": "movie", "title": "The Shawshank Redemption", "release_year": 1994,
        "genres": ["Drama", "Crime"], "director_or_creator": "Frank Darabont",
        "cast_top": ["Tim Robbins", "Morgan Freeman", "Bob Gunton", "William Sadler"],
        "poster_path": "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg", "vote_average": 8.7, "vote_count": 27000, "popularity": 165.0,
        "overview": "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden."
    },
    {
        "tmdb_id": 238, "media_type": "movie", "title": "The Godfather", "release_year": 1972,
        "genres": ["Drama", "Crime"], "director_or_creator": "Francis Ford Coppola",
        "cast_top": ["Marlon Brando", "Al Pacino", "James Caan", "Robert Duvall"],
        "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg", "vote_average": 8.7, "vote_count": 20400, "popularity": 140.0,
        "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers."
    },
    {
        "tmdb_id": 680, "media_type": "movie", "title": "Pulp Fiction", "release_year": 1994,
        "genres": ["Thriller", "Crime"], "director_or_creator": "Quentin Tarantino",
        "cast_top": ["John Travolta", "Samuel L. Jackson", "Uma Thurman", "Bruce Willis"],
        "poster_path": "/vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg", "vote_average": 8.5, "vote_count": 27500, "popularity": 125.0,
        "overview": "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper. Their adventures unfurl in three stories that ingeniously trip back and forth in time."
    },
    {
        "tmdb_id": 496243, "media_type": "movie", "title": "Parasite", "release_year": 2019,
        "genres": ["Comedy", "Thriller", "Drama"], "director_or_creator": "Bong Joon-ho",
        "cast_top": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong", "Choi Woo-shik"],
        "poster_path": "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg", "vote_average": 8.5, "vote_count": 18000, "popularity": 105.0,
        "overview": "All unemployed, Ki-taek's family takes peculiar interest in the wealthy and glamorous Parks for their livelihood until they get entangled in an unexpected incident."
    },
    {
        "tmdb_id": 244786, "media_type": "movie", "title": "Whiplash", "release_year": 2014,
        "genres": ["Drama", "Music"], "director_or_creator": "Damien Chazelle",
        "cast_top": ["Miles Teller", "J.K. Simmons", "Paul Reiser", "Melissa Benoist"],
        "poster_path": "/7fn624j5lj3xTme2SgiLCeuedmO.jpg", "vote_average": 8.4, "vote_count": 14700, "popularity": 88.0,
        "overview": "Under the direction of a ruthless instructor, a talented young drummer begins to pursue perfection at any cost, pushing himself to the brink of both his ability and his sanity."
    },
    {
        "tmdb_id": 11324, "media_type": "movie", "title": "Shutter Island", "release_year": 2010,
        "genres": ["Drama", "Thriller", "Mystery"], "director_or_creator": "Martin Scorsese",
        "cast_top": ["Leonardo DiCaprio", "Mark Ruffalo", "Ben Kingsley", "Max von Sydow"],
        "poster_path": "/nrmXQ0zcZUL8jFLrakWc90IR8z9.jpg", "vote_average": 8.2, "vote_count": 23500, "popularity": 95.0,
        "overview": "World War II soldier-turned-U.S. Marshal Teddy Daniels investigates the disappearance of a patient from Boston's Shutter Island Ashecliffe Hospital, uncovering sinister truths about the facility."
    },
    {
        "tmdb_id": 77, "media_type": "movie", "title": "Memento", "release_year": 2000,
        "genres": ["Mystery", "Thriller"], "director_or_creator": "Christopher Nolan",
        "cast_top": ["Guy Pearce", "Carrie-Anne Moss", "Joe Pantoliano", "Mark Boone Junior"],
        "poster_path": "/nzlv62aC0octS5AklAiWpXLX9Z0.jpg", "vote_average": 8.2, "vote_count": 14500, "popularity": 55.0,
        "overview": "Leonard Shelby is tracking down the man who raped and murdered his wife. The difficulty of locating his wife's killer, however, is compounded by the fact that he suffers from a rare, untreatable form of short-term memory loss."
    },
    {
        "tmdb_id": 1422, "media_type": "movie", "title": "The Departed", "release_year": 2006,
        "genres": ["Drama", "Thriller", "Crime"], "director_or_creator": "Martin Scorsese",
        "cast_top": ["Leonardo DiCaprio", "Matt Damon", "Jack Nicholson", "Mark Wahlberg"],
        "poster_path": "/nT97ifVT2J1yMQmeq20Qblg61T.jpg", "vote_average": 8.2, "vote_count": 14500, "popularity": 75.0,
        "overview": "To take down South Boston's Irish Mafia, the police send in one of their own to infiltrate the underworld, not realizing the syndicate has done likewise."
    },
    {
        "tmdb_id": 87101, "media_type": "movie", "title": "Oppenheimer", "release_year": 2023,
        "genres": ["Drama", "History"], "director_or_creator": "Christopher Nolan",
        "cast_top": ["Cillian Murphy", "Emily Blunt", "Matt Damon", "Robert Downey Jr."],
        "poster_path": "/oZRVDpNtmHk8M1VYy1aeOWUXgbC.jpg", "vote_average": 8.1, "vote_count": 9200, "popularity": 130.0,
        "overview": "The story of J. Robert Oppenheimer's role in the development of the atomic bomb during World War II, examining the political, personal, and moral ramifications of his creation."
    },
    {
        "tmdb_id": 155, "media_type": "movie", "title": "The Dark Knight", "release_year": 2008,
        "genres": ["Drama", "Action", "Crime", "Thriller"], "director_or_creator": "Christopher Nolan",
        "cast_top": ["Christian Bale", "Heath Ledger", "Michael Caine", "Gary Oldman"],
        "poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg", "vote_average": 8.5, "vote_count": 32000, "popularity": 130.0,
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker."
    },
    # TV Series Masterpieces
    {
        "tmdb_id": 1396, "media_type": "tv", "title": "Breaking Bad", "release_year": 2008,
        "genres": ["Drama", "Crime"], "director_or_creator": "Vince Gilligan",
        "cast_top": ["Bryan Cranston", "Aaron Paul", "Anna Gunn", "Dean Norris"],
        "poster_path": "/anFx9aTOOYqgS3v7x3R84Kz67ly.jpg", "vote_average": 8.9, "vote_count": 14200, "popularity": 220.0,
        "overview": "Walter White, a New Mexico chemistry teacher diagnosed with terminal lung cancer, teams up with a former student to cook and sell methamphetamine to secure his family's financial future before he dies."
    },
    {
        "tmdb_id": 60059, "media_type": "tv", "title": "Better Call Saul", "release_year": 2015,
        "genres": ["Crime", "Drama"], "director_or_creator": "Vince Gilligan, Peter Gould",
        "cast_top": ["Bob Odenkirk", "Rhea Seehorn", "Jonathan Banks", "Patrick Fabian"],
        "poster_path": "/zjg4jpK1Wp2kiRvtt5ND0kznako.jpg", "vote_average": 8.7, "vote_count": 5200, "popularity": 110.0,
        "overview": "Six years before he begins to represent Albuquerque's most notorious criminal, Jimmy McGill is a small-time attorney hustling to champion his underdog clients and make a name for himself."
    },
    {
        "tmdb_id": 1399, "media_type": "tv", "title": "Game of Thrones", "release_year": 2011,
        "genres": ["Sci-Fi & Fantasy", "Drama", "Action & Adventure"], "director_or_creator": "David Benioff, D.B. Weiss",
        "cast_top": ["Peter Dinklage", "Kit Harington", "Emilia Clarke", "Lena Headey"],
        "poster_path": "/1XS1oqL89opfnbLl8WnZY1O1uJx.jpg", "vote_average": 8.4, "vote_count": 23500, "popularity": 210.0,
        "overview": "Seven noble families fight for control of the mythical land of Westeros. Friction between the houses leads to full-scale war. All while a very ancient evil awakens in the farthest north."
    },
    {
        "tmdb_id": 87108, "media_type": "tv", "title": "Chernobyl", "release_year": 2019,
        "genres": ["Drama", "History"], "director_or_creator": "Craig Mazin",
        "cast_top": ["Jared Harris", "Stellan Skarsgård", "Emily Watson", "Paul Ritter"],
        "poster_path": "/hlLXt2tOPT6RRnjiUmoxyG1LTFi.jpg", "vote_average": 8.7, "vote_count": 6100, "popularity": 85.0,
        "overview": "The true story of one of the worst man-made catastrophes in history: the catastrophic nuclear accident at Chernobyl. A tale of the brave men and women who sacrificed to save Europe from unimaginable disaster."
    },
    {
        "tmdb_id": 94997, "media_type": "tv", "title": "House of the Dragon", "release_year": 2022,
        "genres": ["Drama", "Action & Adventure", "Sci-Fi & Fantasy"], "director_or_creator": "Ryan J. Condal, George R.R. Martin",
        "cast_top": ["Matt Smith", "Emma D'Arcy", "Olivia Cooke", "Rhys Ifans"],
        "poster_path": "/7V0Ebks0GgpKvQ7QbLAIdX5dos4.jpg", "vote_average": 8.4, "vote_count": 4800, "popularity": 160.0,
        "overview": "The Targaryen dynasty is at the absolute apex of its power, with more than 15 dragons under their yoke. Most empires crumble from such heights. In the case of the Targaryens, their slow fall begins when King Viserys breaks with a century of tradition by naming his daughter Rhaenyra heir to the Iron Throne."
    },
    {
        "tmdb_id": 76479, "media_type": "tv", "title": "The Boys", "release_year": 2019,
        "genres": ["Sci-Fi & Fantasy", "Action & Adventure"], "director_or_creator": "Eric Kripke",
        "cast_top": ["Karl Urban", "Jack Quaid", "Antony Starr", "Erin Moriarty"],
        "poster_path": "/in1R2dDc421JxsoRWaIIAqVI2KE.jpg", "vote_average": 8.4, "vote_count": 10200, "popularity": 170.0,
        "overview": "A fun and irreverent take on what happens when superheroes—who are as popular as celebrities, as influential as politicians, and as revered as gods—abuse their superpowers rather than use them for good. Intent on stopping the corrupt superheroes, a group of vigilantes known informally as 'The Boys' embark on a heroic quest to expose the truth."
    },
    {
        "tmdb_id": 66732, "media_type": "tv", "title": "Stranger Things", "release_year": 2016,
        "genres": ["Sci-Fi & Fantasy", "Drama", "Mystery"], "director_or_creator": "The Duffer Brothers",
        "cast_top": ["Millie Bobby Brown", "Finn Wolfhard", "Winona Ryder", "David Harbour"],
        "poster_path": "/uOOtwVbSr4QDjAGIifLDwpb2Pdl.jpg", "vote_average": 8.6, "vote_count": 17000, "popularity": 195.0,
        "overview": "When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces and one strange little girl."
    },
    {
        "tmdb_id": 119051, "media_type": "tv", "title": "Wednesday", "release_year": 2022,
        "genres": ["Sci-Fi & Fantasy", "Mystery", "Comedy"], "director_or_creator": "Alfred Gough, Miles Millar",
        "cast_top": ["Jenna Ortega", "Gwendoline Christie", "Riki Lindhome", "Jamie McShane"],
        "poster_path": "/9PFonBhy4cQy7Jz20NpMygczOkv.jpg", "vote_average": 8.4, "vote_count": 8300, "popularity": 145.0,
        "overview": "A sleuthing, supernaturally infused mystery charting Wednesday Addams' years as a student at Nevermore Academy, as she attempts to master her emerging psychic ability, thwart a monstrous killing spree, and solve the murder mystery that embroiled her parents 25 years ago."
    },
    {
        "tmdb_id": 100088, "media_type": "tv", "title": "The Last of Us", "release_year": 2023,
        "genres": ["Drama", "Sci-Fi & Fantasy", "Action & Adventure"], "director_or_creator": "Craig Mazin, Neil Druckmann",
        "cast_top": ["Pedro Pascal", "Bella Ramsey", "Gabriel Luna", "Anna Torv"],
        "poster_path": "/dmo6TYuuJgaYinXBPjrgG9mB5od.jpg", "vote_average": 8.6, "vote_count": 5200, "popularity": 135.0,
        "overview": "Twenty years after modern civilization has been destroyed, Joel, a hardened survivor, is hired to smuggle Ellie, a 14-year-old girl, out of an oppressive quarantine zone. What starts as a small job soon becomes a brutal, heartbreaking journey, as they both must traverse the U.S. and depend on each other for survival."
    },
    {
        "tmdb_id": 70523, "media_type": "tv", "title": "Dark", "release_year": 2017,
        "genres": ["Sci-Fi & Fantasy", "Drama", "Mystery"], "director_or_creator": "Baran bo Odar, Jantje Friese",
        "cast_top": ["Louis Hofmann", "Oliver Masucci", "Jördis Triebel", "Maja Schöne"],
        "poster_path": "/apbrbWs8M9lyOpJYU5WXrpFbk1Z.jpg", "vote_average": 8.4, "vote_count": 6200, "popularity": 95.0,
        "overview": "A missing child sets four families on a frantic hunt for answers as they unearth a mind-bending mystery that spans three generations in a small German town."
    },
    {
        "tmdb_id": 85552, "media_type": "tv", "title": "Euphoria", "release_year": 2019,
        "genres": ["Drama"], "director_or_creator": "Sam Levinson",
        "cast_top": ["Zendaya", "Hunter Schafer", "Sydney Sweeney", "Jacob Elordi"],
        "poster_path": "/ypmtwojDd751Peszi62DVLytqqC.jpg", "vote_average": 8.3, "vote_count": 9400, "popularity": 120.0,
        "overview": "A group of high school students navigate love and friendships in a world of drugs, sex, trauma, and social media."
    },
    {
        "tmdb_id": 71446, "media_type": "tv", "title": "Money Heist", "release_year": 2017,
        "genres": ["Crime", "Drama"], "director_or_creator": "Álex Pina",
        "cast_top": ["Úrsula Corberó", "Álvaro Morte", "Itziar Ituño", "Pedro Alonso"],
        "poster_path": "/reEMJA1uzscCbkpeRJeTT2bjqUp.jpg", "vote_average": 8.2, "vote_count": 18200, "popularity": 115.0,
        "overview": "To carry out the biggest heist in history, a mysterious man called The Professor recruits a band of eight robbers who have a single characteristic: none of them has anything to lose."
    },
    # Animation & Studio Ghibli & Pixar
    {
        "tmdb_id": 129, "media_type": "movie", "title": "Spirited Away", "release_year": 2001,
        "genres": ["Animation", "Family", "Fantasy"], "director_or_creator": "Hayao Miyazaki",
        "cast_top": ["Rumi Hiiragi", "Miyu Irino", "Mari Natsuki", "Takashi Naito"],
        "poster_path": "/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg", "vote_average": 8.5, "vote_count": 16200, "popularity": 95.0,
        "overview": "A young girl, Chihiro, becomes trapped in a strange new world of spirits. When her parents undergo a mysterious transformation, she must call upon the courage she never knew she had to free her family."
    },
    {
        "tmdb_id": 493529, "media_type": "movie", "title": "Spider-Man: Into the Spider-Verse", "release_year": 2018,
        "genres": ["Action", "Adventure", "Animation", "Science Fiction"], "director_or_creator": "Bob Persichetti, Peter Ramsey",
        "cast_top": ["Shameik Moore", "Jake Johnson", "Hailee Steinfeld", "Mahershala Ali"],
        "poster_path": "/v7UF7ypAqjsFZFdjksjQ7IUpXdn.jpg", "vote_average": 8.4, "vote_count": 15000, "popularity": 95.0,
        "overview": "Teen Miles Morales becomes the new Spider-Man, while crossing paths with five counterparts from other dimensions to stop a threat for all realities."
    },
    {
        "tmdb_id": 569094, "media_type": "movie", "title": "Spider-Man: Across the Spider-Verse", "release_year": 2023,
        "genres": ["Animation", "Action", "Adventure", "Science Fiction"], "director_or_creator": "Joaquim Dos Santos, Kemp Powers",
        "cast_top": ["Shameik Moore", "Hailee Steinfeld", "Oscar Isaac", "Jake Johnson"],
        "poster_path": "/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg", "vote_average": 8.4, "vote_count": 6800, "popularity": 130.0,
        "overview": "After reuniting with Gwen Stacy, Brooklyn's full-time, friendly neighborhood Spider-Man is catapulted across the Multiverse, where he encounters the Spider Society, a team of Spider-People charged with protecting the Multiverse's very existence."
    },
    {
        "tmdb_id": 372058, "media_type": "movie", "title": "Your Name.", "release_year": 2016,
        "genres": ["Animation", "Romance", "Drama"], "director_or_creator": "Makoto Shinkai",
        "cast_top": ["Ryunosuke Kamiki", "Mone Kamishiraishi", "Ryo Narita", "Aoi Yuki"],
        "poster_path": "/vfJFJPepRKapMd5G2ro7klIRysq.jpg", "vote_average": 8.5, "vote_count": 11000, "popularity": 80.0,
        "overview": "High schoolers Mitsuha and Taki are complete strangers living separate lives in different parts of Japan. But when they suddenly begin swapping bodies in their dreams, a bizarre connection forms between them that defies space and time."
    }
]

def export():
    output_path = Path(__file__).resolve().parent.parent / "data" / "starter_catalog.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(titles)} curated titles to {output_path}")

if __name__ == "__main__":
    export()
