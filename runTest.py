from subprocess import call
numRuns = 100;
cr = 0

while cr < numRuns:
  call(["./TEST_SUITE"])
  cr += 1
