# Document-Search-Engine
Indexer of documents that supports a boolean query language.

## Utilization
Once in the indexer directory:
```
$ python indexer.py <corpus-directory> <stop-words-file>
```
Then a progress bar will appear showing the status of the indexing process.
When the indexing finishes, you will be asked to enter a query.
The query must be formed by words separated by any of the three supported operands: "or", "and", "not".
The names of the documents resulting from the query will be shown in the console and you will be asked for another query until you enter "--exit".

## Dependencies
 * Python 3
 * tqdm
