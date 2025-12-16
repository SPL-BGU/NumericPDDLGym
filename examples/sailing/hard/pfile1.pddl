;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_3_9)

	(:domain sailing)

	(:objects
		b0 b1 b2 - boat
		p0 p1 p2 p3 p4 p5 p6 p7 p8 - person
	)

  (:init
		(= (x b0) 7)
(= (y b0) 0)
(= (x b1) 8)
(= (y b1) 0)
(= (x b2) 9)
(= (y b2) 0)

		(= (d p0) 0)
(= (d p1) 0)
(= (d p2) 1)
(= (d p3) 0)
(= (d p4) 0)
(= (d p5) 1)
(= (d p6) 1)
(= (d p7) 0)
(= (d p8) 2)

	)

	(:goal
		(and
			(saved p0)
(saved p1)
(saved p2)
(saved p3)
(saved p4)
(saved p5)
(saved p6)
(saved p7)
(saved p8)
		)
	)
)

