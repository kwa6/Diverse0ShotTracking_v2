
import ezpyzy as ez

# string using the pound sign
my_string = 'I was in England and bought an apple for £1.50'
print(my_string.encode('ascii'))

ez.File('encoding_issue.txt').save(my_string)
print(ez.File('encoding_issue.txt').load())

