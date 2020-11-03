# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:47:36 2020

@author: janit
"""
import joblib

def load_model():
  model = joblib.load('SVMclassifier.pkl')
  return model

def main():
  SVMmodel = load_model
  return SVMmodel

if __name__ == "__main__":
    SVMclassifier = main()  
    
    