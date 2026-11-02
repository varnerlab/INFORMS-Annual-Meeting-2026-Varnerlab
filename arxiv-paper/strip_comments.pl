#!/usr/bin/perl
# Strip LaTeX comments, in place, from the copies of the .tex files that go into the arXiv archive.
# The repository sources keep their comments (they name the result file behind each number).
# A line that holds only a comment is deleted. A comment at the end of a line is cut back to its "%",
# which keeps the end-of-line suppression the "%" was there for. An escaped "\%" is left alone.
use strict;
use warnings;

$^I = '';
while (my $line = <>) {
    # Delete a line that is only a comment.
    if ($line =~ /^\s*%/) {
        next;
    }
    # Cut a trailing comment back to its "%".
    $line =~ s/(?<!\\)%.*$/%/;
    print $line;
}
