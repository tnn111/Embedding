
# New requests

Look at the calculate_kmer_frequencies code. I'd like to make some changes to the code:

- Start each line with the ID from the FASTA defline.
- Then output the length of the nucleotide sequence.
- Output all of the normalized 7-mers frequencies as before.
- Then output all of the normalized 4-mer frequencies. Remember that 4-mers can have palindromes so make sure you count well.
- Then output all of the normalized 3-mer frequencies.
- Finally output the GC content of the sequence as a floating point number between 0 and 1.

For 3-mers, 4-mers and 7-mers make sure you count canonical sequences.

Ask me any questions you like before doing it.

# Answers

1. Just the first word after the '>' character. Stop at white space or the end of the line.
2. Use the sequence length after filtering to only ATGC bases.
3. Normalize 3-mers, 4-mers and 7-mers separately.
4. Your example is correct. Separate fields with a space.
5. You are correct regarding counting canonical 4-mers.