import React from 'react'

interface OnBoardingProgressBarProps {
  allTextArray: React.JSX.Element[]
  index: number
  setIndex: React.Dispatch<React.SetStateAction<number>>
}

const OnBoardingProgressBar = ({ allTextArray, index, setIndex }: OnBoardingProgressBarProps) => {
  return (
    <div className="flex justify-center gap-x-4">
      <div className="mr-2">Logo 42</div>
      {allTextArray.map((_, i) => (
        <div
          key={i}
          className={`w-2 h-2 tablet:w-4 tablet:h-4 rounded-full cursor-pointer ${
            i === index ? 'bg-cyan' : 'bg-gray-400'
          }`}
          onClick={() => setIndex(i)}
        />
      ))}
      <div className="ml-2">Logo Lab</div>
    </div>
  )
}

export default OnBoardingProgressBar
