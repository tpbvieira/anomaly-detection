import csv
import pandas
game = pandas.read_csv("data.csv")
game = pandas.get_dummies(game)
game.to_csv("main_data.csv", sep=',', encoding='utf-8')

