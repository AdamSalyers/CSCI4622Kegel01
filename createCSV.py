ident = traindf['id'].values
identarr = np.array(ident)
predarr = np.array(predictions)
df = pd.DataFrame({'id':identarr, 'price':predarr})
df.head()



df.to_csv('kaggle_carpenter_salyers.csv', index=False)
