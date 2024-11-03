go :- light(green), not carsIncoming.
go :- light(green), turnSignalOn.
go :- light(yellow), not carsIncoming, turnSignalOn.
:- light(red), go.
:- light(yellow), carsIncoming, go.
:- light(yellow), not turnSignalOn, go.


light(green).
-carsIncoming.
turnSignalOn.

carsIncoming :- not -carsIncoming.
turnSignalOn :- not -turnSignalOn.