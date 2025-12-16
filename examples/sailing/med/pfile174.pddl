;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_2_4)

	(:domain sailing)

	(:objects
		b0 b1 - boat
		p0 p1 p2 p3 - person
	)

  (:init
		(= (x b0) 8)
(= (y b0) 0)
(= (x b1) -4)
(= (y b1) 0)

		(= (d p0) 1)
(= (d p1) 0)
(= (d p2) 0)
(= (d p3) 2)

	)

	(:goal
		(and
			(saved p0)
(saved p1)
(saved p2)
(saved p3)
		)
	)
)

